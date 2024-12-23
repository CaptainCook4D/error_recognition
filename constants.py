class Constants:
    RESULTS = "results"
    RESULT_ID = "result_id"
    DATE = "date"
    RESULT_DETAILS = "result_details"
    BEST_MODEL_TYPE = "best_model_type"

    ACCURACY = "accuracy"
    LOSS = "loss"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    PR_AUC = "pr_auc"

    STEP_METRICS = "step_metrics"
    SUB_STEP_METRICS = "sub_step_metrics"

    TASK_NAME = "task_name"
    VARIANT = "variant"
    MODEL_NAME = "model_name"
    BACKBONE = "backbone"
    MODALITY = "modality"
    SPLIT = "split"

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    ERROR_RECOGNITION = "error_recognition"
    ERROR_CATEGORY_RECOGNITION = "error_category_recognition"
    EARLY_ERROR_RECOGNITION = "early_error_recognition"

    RECORDINGS_SPLIT = "recordings"
    PERSON_SPLIT = "person"
    ENVIRONMENT_SPLIT = "environment"
    STEP_SPLIT = "step"

    # --------------------- MODEL SPECIFIC CONSTANTS ---------------------
    OMNIVORE = "omnivore"
    RESNET3D = "3dresnet"
    X3D = "x3d"
    SLOWFAST = "slowfast"
    IMAGEBIND = "imagebind"

    IMAGEBIND_VIDEO = "imagebind_video"
    IMAGEBIND_AUDIO = "imagebind_audio"
    IMAGEBIND_TEXT = "imagebind_text"
    IMAGEBIND_DEPTH = "imagebind_depth"

    MLP_VARIANT = "MLP"
    TRANSFORMER_VARIANT = "Transformer"
    MULTIMODAL_VARIANT = "Multimodal"

    # ----------------------- WANDB CONSTANTS -----------------------
    WANDB_PROJECT = "error_recognition"

    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    DEPTH = "depth"

    # ---------------------- ERROR CATEGORY TYPES ----------------------
    TECHNIQUE_ERROR = "Technique Error"
    PREPARATION_ERROR = "Preparation Error"
    TEMPERATURE_ERROR = "Temperature Error"
    MEASUREMENT_ERROR = "Measurement Error"
    TIMING_ERROR = "Timing Error"

