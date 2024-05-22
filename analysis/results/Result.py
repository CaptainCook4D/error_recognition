from datetime import datetime
import uuid
from typing import Dict
from constants import Constants as const


class Metrics:

    def __init__(
            self,
            accuracy: float,
            precision: float,
            recall: float,
            f1: float,
            auc: float
    ):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.auc = auc

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, float]) -> 'Metrics':
        return cls(
            accuracy=metrics_dict[const.ACCURACY],
            precision=metrics_dict[const.PRECISION],
            recall=metrics_dict[const.RECALL],
            f1=metrics_dict[const.F1],
            auc=metrics_dict[const.AUC]
        )

    def to_dict(self) -> Dict:
        return {
            const.ACCURACY: self.accuracy,
            const.PRECISION: self.precision,
            const.RECALL: self.recall,
            const.F1: self.f1,
            const.AUC: self.auc
        }

class ResultDetails:

    def __init__(
            self,
            sub_step_metrics,
            step_metrics
    ):
        self.sub_step_metrics = sub_step_metrics
        self.step_metrics = step_metrics

    def add_sub_step_metrics(self, sub_step_metrics):
        self.sub_step_metrics = sub_step_metrics

    def add_step_metrics(self, step_metrics):
        self.step_metrics = step_metrics

    def to_dict(self) -> Dict:
        result_details_dict = {}
        if self.sub_step_metrics is not None:
            sub_step_metrics = self.sub_step_metrics.to_dict()
            result_details_dict[const.SUB_STEP_METRICS] = sub_step_metrics

        if self.step_metrics is not None:
            step_metrics = self.step_metrics.to_dict()
            result_details_dict[const.STEP_METRICS] = step_metrics

        return result_details_dict

    @classmethod
    def from_dict(cls, result_details_dict: Dict) -> 'ResultDetails':
        sub_step_metrics = None
        step_metrics = None

        if const.SUB_STEP_METRICS in result_details_dict:
            sub_step_metrics = Metrics.from_dict(result_details_dict[const.SUB_STEP_METRICS])

        if const.STEP_METRICS in result_details_dict:
            step_metrics = Metrics.from_dict(result_details_dict[const.STEP_METRICS])

        return cls(
            sub_step_metrics=sub_step_metrics,
            step_metrics=step_metrics
        )


class Result:

    def __init__(
            self,
            task_name,
            variant,
            backbone,
            modality,
            split,
            model_name=None,
            result_id=None
    ):
        """
            task_name="ErrorRecognition",
            variant="MLP",
            backbone="Omnivore",
        """
        self.task_name = task_name
        self.variant = variant
        self.backbone = backbone
        self.modality = modality
        self.split = split
        self.model_name = model_name
        self.result_id = result_id

        if self.model_name is None:
            self.model_name = f"{self.task_name}_{self.variant}_{self.backbone}_{self.modality}_{self.split}"

        # Specific Attributes
        self.best_model_type = None

        # Common Attributes
        self.result_details = None

        if result_id is None:
            self.result_id = str(uuid.uuid4())
        else:
            self.result_id = result_id

        self.result_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_result_details(self, result_details: ResultDetails):
        self.result_details = result_details

    def to_dict(self) -> Dict:
        result_dict = {
            const.TASK_NAME: self.task_name,
            const.VARIANT: self.variant,
            const.MODEL_NAME: self.model_name,
            const.BACKBONE: self.backbone,
            const.SPLIT: self.split,
            const.MODALITY: self.modality,
            const.RESULT_ID: self.result_id,
            const.DATE: self.result_date
        }

        if self.result_details is not None:
            result_details = self.result_details.to_dict()
            result_dict[const.RESULT_DETAILS] = result_details

        if self.best_model_type is not None:
            result_dict[const.BEST_MODEL_TYPE] = self.best_model_type

        return result_dict

    @classmethod
    def from_dict(cls, result_dict: Dict) -> 'Result':
        result = cls(
            task_name=result_dict[const.TASK_NAME],
            variant=result_dict[const.VARIANT],
            backbone=result_dict[const.BACKBONE],
            modality=result_dict[const.MODALITY],
            split=result_dict[const.SPLIT],
            model_name=result_dict[const.MODEL_NAME],
            result_id=result_dict[const.RESULT_ID]
        )

        result_details = None
        if 'result_details' in result_dict:
            result_details = ResultDetails.from_dict(result_dict['result_details'])
        result.add_result_details(result_details)

        best_model_type = None
        if 'best_model_type' in result_dict:
            best_model_type = result_dict['best_model_type']
        result.best_model_type = best_model_type

        return result
