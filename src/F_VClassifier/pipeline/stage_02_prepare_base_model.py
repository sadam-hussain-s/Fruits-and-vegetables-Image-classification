from F_VClassifier.config.configuration import ConfigurationManager
from F_VClassifier.components.prepare_base_model import PrepareBaseModel
from F_VClassifier import logger

"""
The VGG-16 model is a convolutional neural network (CNN) architecture 
that was proposed by the Visual Geometry Group (VGG) at the University of Oxford. 
It is characterized by its depth, consisting of 16 layers, 
including 13 convolutional layers and 3 fully connected layers. 
VGG-16 is renowned for its simplicity and effectiveness, as well as its ability 
to achieve strong performance on various computer vision tasks, 
including image classification and object recognition.
"""

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e