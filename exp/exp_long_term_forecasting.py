from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, MultiMetricsCalculator
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
warnings.filterwarnings('ignore')
import logging
from collections import Counter


def _create_versioned_dirs_and_move_log(base_model_dir, base_logs_dir, tmp_log_path, logger=None, formatter=None):
    """Create next vXX dirs under base_model_dir/base_logs_dir, move tmp_log_path to final
    logs_dir and return (model_dir, logs_dir, final_log_path).
    If logger and formatter are provided, reattach FileHandler to the logger pointing to final_log_path.
    """
    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir, exist_ok=True)
    if not os.path.exists(base_logs_dir):
        os.makedirs(base_logs_dir, exist_ok=True)

    # compute next version by inspecting both model and log dirs so they stay in sync
    def _versions_in(dir_path):
        if not os.path.exists(dir_path):
            return []
        names = [n for n in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, n))]
        vers = []
        for n in names:
            if n.startswith('v'):
                try:
                    vers.append(int(n[1:]))
                except Exception:
                    continue
        return vers

    vers_model = _versions_in(base_model_dir)
    vers_logs = _versions_in(base_logs_dir)
    all_vers = vers_model + vers_logs
    next_v = max(all_vers) + 1 if all_vers else 1
    model_dir = os.path.join(base_model_dir, f'v{next_v:02d}')
    logs_dir = os.path.join(base_logs_dir, f'v{next_v:02d}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    final_log_path = os.path.join(logs_dir, 'train_log.txt')
    try:
        import shutil
        shutil.move(tmp_log_path, final_log_path)
    except Exception:
        # If move fails, try copying
        try:
            shutil.copy(tmp_log_path, final_log_path)
        except Exception:
            pass

    # reattach file handler to logger if requested
    if logger is not None and formatter is not None:
        try:
            # remove any existing file handlers
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    try:
                        logger.removeHandler(h)
                        h.close()
                    except Exception:
                        pass
            fh = logging.FileHandler(final_log_path, mode='a', encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            pass

    return model_dir, logs_dir, final_log_path


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc='Validation', total=len(vali_loader), unit='batch')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
     
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # On Windows, avoid spawning worker processes which can trigger
        # multiprocessing import issues; force single-process data loading.
        if sys.platform.startswith('win'):
            self.args.num_workers = 0

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create legacy checkpoint path and new versioned model/log directories
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # create versioned directories for trained model and logs
        base_model_dir = os.path.join('modelos_treinados', setting)
        base_logs_dir = os.path.join('logs', setting)
        # determine next version number by scanning existing dirs
        def _next_version(base_dir):
            if not os.path.exists(base_dir):
                return 1
            names = [n for n in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, n))]
            vers = []
            for n in names:
                if n.startswith('v'):
                    try:
                        vers.append(int(n[1:]))
                    except Exception:
                        continue
            return max(vers) + 1 if vers else 1

        # We'll create versioned model/log directories lazily after the first epoch
        # so we avoid creating unused vXX folders. First, set up logging to
        # stdout and to a temporary log file under `logs/<setting>/tmp/train_log.txt`.
        logger = logging.getLogger('train_logger')
        logger.setLevel(logging.INFO)
        # remove existing handlers
        if logger.handlers:
            for h in list(logger.handlers):
                try:
                    logger.removeHandler(h)
                    h.close()
                except Exception:
                    pass

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # temp logs dir
        tmp_logs_dir = os.path.join(base_logs_dir, 'tmp')
        os.makedirs(tmp_logs_dir, exist_ok=True)
        fh_path = os.path.join(tmp_logs_dir, 'train_log.txt')
        fh = logging.FileHandler(fh_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # lazy creation state
        version_created = False
        model_dir = None
        logs_dir = None

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # Visual progress bar for training loop
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1} Train', total=train_steps, unit='batch')):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device).squeeze(0)

                batch_y = batch_y.float().to(self.device).squeeze(0)
                batch_x_mark = batch_x_mark.float().to(self.device).squeeze(0)
                batch_y_mark = batch_y_mark.float().to(self.device).squeeze(0)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # import pdb
                    # pdb.set_trace()
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info("iters: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, train_steps, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # Validation with progress bar
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # On first epoch end, create versioned directories and move temp log
            if not version_created:
                v_model = _next_version(base_model_dir)
                v_logs = _next_version(base_logs_dir)
                model_dir = os.path.join(base_model_dir, f'v{v_model:02d}')
                logs_dir = os.path.join(base_logs_dir, f'v{v_logs:02d}')
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(logs_dir, exist_ok=True)

                # move temporary log file into the new logs_dir
                final_log_path = os.path.join(logs_dir, 'train_log.txt')
                try:
                    import shutil
                    # flush and close the current file handler before moving
                    for h in list(logger.handlers):
                        if isinstance(h, logging.FileHandler):
                            try:
                                h.flush()
                                h.close()
                            except Exception:
                                pass
                            try:
                                logger.removeHandler(h)
                            except Exception:
                                pass
                    # move temp log
                    shutil.move(fh_path, final_log_path)
                    # attach new FileHandler pointing to final log
                    fh = logging.FileHandler(final_log_path, mode='a', encoding='utf-8')
                    fh.setLevel(logging.INFO)
                    fh.setFormatter(formatter)
                    logger.addHandler(fh)
                    fh_path = final_log_path
                except Exception:
                    # if move fails, keep logging to temp
                    pass
                version_created = True

            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # pass optimizer and epoch and a full checkpoint directory to EarlyStopping
            early_stopping(vali_loss, self.model, path, optimizer=model_optim, epoch=epoch + 1, extra_dir=model_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        # load best model weights (legacy) into the model
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # also copy full checkpoint if exists into model_dir for provenance
        full_ckpt_src = os.path.join(path, 'checkpoint_full.pth')
        if os.path.exists(full_ckpt_src):
            try:
                import shutil
                shutil.copy(full_ckpt_src, os.path.join(model_dir, 'checkpoint_full.pth'))
            except Exception:
                pass

        # Ensure file handler is flushed and closed so the log file contains the run output
        try:
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    try:
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                    try:
                        logger.removeHandler(h)
                    except Exception:
                        pass
        except Exception:
            pass

        # Explicitly append a final summary line to the log file to avoid empty files
        try:
            with open(fh_path, 'a', encoding='utf-8') as wf:
                wf.write(f"Run completed for setting: {setting}\n")
        except Exception:
            pass

        # Ensure logging subsystem flushes
        try:
            logging.shutdown()
        except Exception:
            pass

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        import pdb
        pdb.set_trace()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metric_multi = MultiMetricsCalculator()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), \
                desc="Calculating metrics", total=len(test_loader), unit="site"):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                metric_multi.update(pred, true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pred = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pred, os.path.join(folder_path, str(i) + '.pdf'))
        avg_mae, avg_mse = metric_multi.get_metrics()
        # creat a dataFrame to save mutli-metrics
        metrics_df = pd.DataFrame({'Variable': ['Temperature', 'Humidity', 'Wind Speed', 'Pressure','Wind Direction'],
                                'MAE': avg_mae.tolist(),
                                'MSE': avg_mse.tolist()})
        print(metrics_df)
        preds = np.array(preds)
        trues = np.array(trues)
        import pdb
        pdb.set_trace()
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
