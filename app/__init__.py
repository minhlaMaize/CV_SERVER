
import logging
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


knn_model = joblib.load("app/model/knn_model.pkl")
svm_model = joblib.load("app/model/svm_model.pkl")
#zLoad scaler
scaler = joblib.load(open("app/model/scaler.pkl",'rb'))
#Load PCA
pca = joblib.load(open("app/model/pca.pkl",'rb'))
