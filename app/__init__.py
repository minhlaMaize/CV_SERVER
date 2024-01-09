
import logging
import joblib
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


model = joblib.load("app/model/model.pkl")


