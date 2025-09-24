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


def _safe_torch_load(path, map_location=None):
    """Robust loader for checkpoints. Some checkpoints are full dicts saved by
    older code and may require `weights_only=False` when loading with newer
    PyTorch. Try the default load first, then retry with `weights_only=False`.
    Returns the loaded object.
    """
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        # Older PyTorch may not accept weights_only kw; re-raise to let caller handle
        raise
    except Exception:
        # Try explicit weights_only=False for PyTorch>=2.6 where default changed
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            # As a last resort, re-raise the original exception
            return torch.load(path, map_location=map_location)


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

        # Warm-start: if an explicit checkpoint path is provided, load compatible
        # parameters into the model before continuing training. This copies only
        # parameters that exist in both state_dicts and have matching shapes.
        if getattr(self.args, 'ckpt_path', None):
            try:
                ck = _safe_torch_load(self.args.ckpt_path, map_location=self.device)
                if isinstance(ck, dict) and 'model_state_dict' in ck:
                    ck_sd = ck['model_state_dict']
                else:
                    ck_sd = ck

                model_sd = self.model.state_dict()
                # build filtered dict
                filtered = {k: v for k, v in ck_sd.items() if k in model_sd and v.size() == model_sd[k].size()}
                if filtered:
                    model_sd.update(filtered)
                    self.model.load_state_dict(model_sd)
                    print(f"Warm-start: loaded {len(filtered)} tensors from checkpoint: {self.args.ckpt_path}")
                else:
                    print("Warm-start: no compatible tensors found in checkpoint; skipping.")
            except Exception as e:
                print("Warm-start load failed:", e)

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

        # Pre-create versioned model/log directories immediately so they exist
        # even if training is interrupted before the end of the first epoch.
        try:
            v_model = _next_version(base_model_dir)
            v_logs = _next_version(base_logs_dir)
            pre_model_dir = os.path.join(base_model_dir, f'v{v_model:02d}')
            pre_logs_dir = os.path.join(base_logs_dir, f'v{v_logs:02d}')
            os.makedirs(pre_model_dir, exist_ok=True)
            os.makedirs(pre_logs_dir, exist_ok=True)
            # create a tmp log file so users see something in logs/<setting>/vXX/
            tmp_log_path = os.path.join(pre_logs_dir, 'train_log.txt')
            open(tmp_log_path, 'a').close()
        except Exception:
            # If any assertion/IO error occurs, continue without failing the run
            pass

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
        # load best model weights (legacy) into the model robustly
        ck = _safe_torch_load(best_model_path, map_location=self.device)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            sd = ck['model_state_dict']
        else:
            sd = ck
        self.model.load_state_dict(sd)

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
            # Allow explicit checkpoint file path via args.ckpt_path (absolute or relative).
            if getattr(self.args, 'ckpt_path', None):
                ckpt_file = self.args.ckpt_path
            else:
                ckpt_file = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')

            ck = _safe_torch_load(ckpt_file, map_location=self.device)
            if isinstance(ck, dict) and 'model_state_dict' in ck:
                sd = ck['model_state_dict']
            else:
                sd = ck
            self.model.load_state_dict(sd)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metric_multi = MultiMetricsCalculator()
        self.model.eval()
        with torch.no_grad():
            # iterate with a readable progress bar
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Testing", total=len(test_loader), unit="batch")):

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
                batch_y_proc = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_np = batch_y_proc.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]

                pred = outputs
                true = batch_y_np

                preds.append(pred)
                trues.append(true)

                metric_multi.update(pred, true)
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pred_vis = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pred_vis, os.path.join(folder_path, str(i) + '.pdf'))

        avg_mae, avg_mse, sedi = metric_multi.get_metrics()
        # sedi has shape (num_thresholds, num_vars); reduce to per-variable by averaging
        try:
            sedi_per_var = np.mean(sedi, axis=0)
        except Exception:
            sedi_per_var = np.zeros_like(avg_mae)

        # create a DataFrame to save multi-metrics (include SEDI averaged per variable)
        # Determine number of variables from avg_mae length and generate names
        try:
            n_vars = len(avg_mae)
        except Exception:
            n_vars = 1
        # If dataset provides variable names via test_data.columns, use them; else default Var1..VarN
        var_names = None
        try:
            if hasattr(test_data, 'columns') and isinstance(test_data.columns, (list, tuple)):
                var_names = test_data.columns
        except Exception:
            var_names = None
        if not var_names:
            var_names = [f'Var{i+1}' for i in range(n_vars)]

        # Ensure arrays match n_vars
        avg_mae = np.array(avg_mae[:n_vars]).tolist()
        avg_mse = np.array(avg_mse[:n_vars]).tolist()
        sedi_per_var = np.array(sedi_per_var[:n_vars]).tolist()

        metrics_df = pd.DataFrame({'Variable': var_names,
                                   'MAE': avg_mae,
                                   'MSE': avg_mse,
                                   'SEDI': sedi_per_var})
        print(metrics_df)

        preds = np.array(preds)
        trues = np.array(trues)
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
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
