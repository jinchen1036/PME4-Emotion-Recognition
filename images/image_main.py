import time
from dotenv import load_dotenv
from utils.settings import Settings

if __name__ == '__main__':
    # set the environment variables
    load_dotenv()

    # set the parameter for audio data
    settings = Settings()
    settings.set_image()

    test_size = 0.3  # for one model
    cv_n = 5  # for cross validation
