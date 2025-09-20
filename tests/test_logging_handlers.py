import logging
import os
import tempfile


def configure_train_logger(log_file_path):
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
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger, fh


def test_filehandler_writes(tmp_path):
    """Ensure logger FileHandler writes to the target file and flush/close works."""
    vdir = tmp_path / "v01"
    vdir.mkdir()
    log_path = vdir / "train_log.txt"

    logger, fh = configure_train_logger(str(log_path))
    logger.info("Test message epoch 1")

    # flush and close handlers
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.flush()
            h.close()
            logger.removeHandler(h)

    # read file and assert content
    assert log_path.exists()
    content = log_path.read_text(encoding='utf-8')
    assert "Test message epoch 1" in content


def test_logger_reconfiguration_writes_both_versions(tmp_path):
    """Simulate v01 then v02 runs and ensure both log files receive their messages."""
    base = tmp_path / "setting"
    v01 = base / "v01"
    v02 = base / "v02"
    v01.mkdir(parents=True)
    v02.mkdir()

    log1 = str(v01 / "train_log.txt")
    logger, fh1 = configure_train_logger(log1)
    logger.info("epoch 1 logged")

    # close first file handler
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.flush()
            h.close()
            logger.removeHandler(h)

    # reconfigure to v02
    log2 = str(v02 / "train_log.txt")
    logger, fh2 = configure_train_logger(log2)
    logger.info("epoch 2 logged")

    # close second
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.flush()
            h.close()
            logger.removeHandler(h)

    # verify both files exist and contain expected messages
    assert (v01 / "train_log.txt").exists()
    assert (v02 / "train_log.txt").exists()
    t1 = (v01 / "train_log.txt").read_text(encoding='utf-8')
    t2 = (v02 / "train_log.txt").read_text(encoding='utf-8')
    assert "epoch 1 logged" in t1
    assert "epoch 2 logged" in t2
