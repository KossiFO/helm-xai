"""Display semantics of the five-method recipe; synthetic attributions only."""
from helm.config import Attribution, Prediction, UserProfile, MethodSelection
from helm.context import HELMContext
from helm.visualization.method_coverage import method_status
from helm.visualization.profile_html import context_html


def test_coverage_distinguishes_empty_failed_and_not_calculated():
    assert method_status('anchors',None,{}) == 'Non calculée'
    assert method_status('anchors',None,{'anchors':'error'}) == 'Échec du calcul'
    cf=Attribution('counterfactual',{},metadata={'counterfactuals':[]})
    assert 'aucune inversion' in method_status('counterfactual',cf,{})
    ig=Attribution('integrated_gradients',{},metadata={'valid':False})
    assert 'contrôles numériques' in method_status('integrated_gradients',ig,{})


def test_ig_probability_units_and_anchor_rule_not_attribution_bars():
    ctx=HELMContext(text='test',model_name='xlmr',user_profile=UserProfile.END_USER)
    ctx.prediction=Prediction('non_toxique',.9,(.9,.1),'xlmr')
    ctx.selected_methods=[MethodSelection('lime',1,1,'test')]
    ig=Attribution('integrated_gradients',{'mot':.2},base_value=.7,convergence_delta=0,
                   metadata={'target_class':0,'target_space':'probability','valid':True,'effective_method':'integrated_gradients','mode':'verified_native'})
    anchor=Attribution('anchors',{'inapplicable':999},metadata={'target_class':0,'anchor_words':['<mot>'],
        'precision':.96,'threshold':.95,'coverage':.2})
    html=context_html(ctx,available={'integrated_gradients':ig,'anchors':anchor})
    assert '-20.00 pts' in html and 'Integrated Gradients (IG)' in html
    assert '96.0%' in html and '&lt;mot&gt;' in html
    assert 'inapplicable' not in html and '+999' not in html
    assert 'complément consultable' in html
    assert 'Coefficients locaux' not in html


def test_real_failure_shape_and_legacy_ig_do_not_claim_verified_gradients():
    assert method_status('integrated_gradients',None,{'integrated_gradients':{'metadata':{'valid':False}}}) == 'Échec des contrôles numériques'
    ctx=HELMContext(text='test',model_name='xlmr',user_profile=UserProfile.END_USER)
    ctx.prediction=Prediction('toxique',.9,(.1,.9),'xlmr')
    for mode,effective,space in [('leave_one_out','loo_replacement','probability'),('captum','integrated_gradients_legacy','logit')]:
        old=Attribution('integrated_gradients',{'mot':.2},metadata={'mode':mode,'effective_method':effective,'target_space':space})
        html=context_html(ctx,available={'integrated_gradients':old})
        assert 'backend historique' in html and 'LOO est une méthode distincte' in html
        assert '+20.00 pts' not in html and 'IG natif : intégration' not in html
