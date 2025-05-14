
class BaseEmbeddingModel():
    """
        Initialize the base embedding model.

        Args:
            model_type (str): The type or name of the model.    
    """
    def __init__(self, model_type):
        pass

    def get_image_features(self, images):
        raise NotImplementedError("get_image_features is not implemented")

    def get_text_features(self, texts):
        raise NotImplementedError("get_text_features is not implemented")