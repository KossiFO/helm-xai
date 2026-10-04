"""Build a portable dashboard from saved Colab calculation objects."""
import joblib
from helm.tabular.profiles import profile_dashboard
from .prepare import ROOT

if __name__ == '__main__':
    reports = {order:{m:joblib.load(ROOT/f'{m}_{order}.joblib') for m in ('shap','lime') if (ROOT/f'{m}_{order}.joblib').exists()} for order in ('highest','lowest')}
    profile_dashboard(reports,title='HELM MAAF — Explorer les scores',group_labels={'highest':'10 scores élevés','lowest':'10 scores faibles'}).to_html(ROOT/'dashboard.html')
