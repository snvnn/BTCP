from btcp.model import trainer


def test_trainer_module_imports_without_ml_dependencies():
    assert trainer.SEQ_LENGTH == 60
    assert trainer.PRED_OFFSETS == [5, 15, 30, 60]


def test_training_paths_are_project_configurable():
    assert "/home/yunh" not in str(trainer.DEFAULT_CSV_PATH)
    assert "/home/yunh" not in str(trainer.DEFAULT_MODEL_DIR)
