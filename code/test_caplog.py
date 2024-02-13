import logging


def test_logging(caplog):
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    # "info" is not recorded by default because the default level is "warning"
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[1].levelname == "ERROR"
