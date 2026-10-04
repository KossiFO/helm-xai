"""Portable HTML dashboards: CSS-only controls, no kernel, JS or CDN dependency."""
from dataclasses import dataclass
from html import escape
from pathlib import Path
from uuid import uuid4
import math

STYLE = """
.helm-dashboard{--navy:#172e52;--blue:#215ac4;--orange:#aa420f;--teal:#006b65;font:15px/1.55 system-ui,-apple-system,sans-serif;color:#243752;background:#f2f6fc;border:1px solid #d4dfef;border-radius:20px;max-width:1180px;margin:12px auto;overflow:clip;overflow-wrap:anywhere}
.helm-dashboard *{box-sizing:border-box}.helm-dashboard .helm-header{background:linear-gradient(120deg,#122440,#234d80);color:white;padding:26px 30px}.helm-dashboard .helm-brand{font-size:13px;font-weight:800;letter-spacing:3px;color:#8de1d4}.helm-dashboard .helm-header h1{font-size:28px;margin:5px 0 3px;line-height:1.2;color:white}.helm-dashboard .helm-header p{margin:0;color:#e0eaff}
.helm-dashboard .helm-controls{position:sticky;top:0;z-index:3;padding:16px 24px;background:#fff;border-bottom:1px solid #d4dfef;box-shadow:0 3px 12px #172e5210}.helm-dashboard .helm-controls fieldset{border:0;margin:0 0 8px;padding:0;display:flex;flex-wrap:wrap;gap:8px}.helm-dashboard legend{font-size:12px;font-weight:800;text-transform:uppercase;color:#435976;margin-bottom:6px}.helm-dashboard label{display:inline-block;padding:9px 17px;border-radius:9px;border:1px solid #bbcce4;background:#f5f8fe;color:#183c6d;font-weight:700;cursor:pointer;min-height:44px}.helm-dashboard label:hover{background:#e4ecff}.helm-dashboard .helm-radio{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden}
.helm-dashboard .helm-panels{padding:24px}.helm-dashboard .helm-view{display:none}.helm-dashboard .helm-view h1{font-size:25px;color:var(--navy);line-height:1.25}.helm-dashboard h2{font-size:20px;color:var(--navy);margin-top:30px}.helm-dashboard h3{font-size:17px;color:#1f497a}.helm-dashboard h4{font-size:15px;color:#385371}.helm-dashboard p{max-width:95ch}.helm-dashboard section{background:white;border:1px solid #d7e2ef;border-radius:14px;padding:20px;margin:16px 0}.helm-dashboard blockquote{border-left:4px solid #24887c;background:#e8f7f3;padding:15px 20px;margin:18px 0;border-radius:0 10px 10px 0;font-size:17px;color:#154c48}
.helm-dashboard details{background:#fff;border:1px solid #d4dfef;border-radius:10px;margin:12px 0;padding:0 16px}.helm-dashboard summary{cursor:pointer;font-weight:700;padding:14px 0;color:#204e88}.helm-dashboard summary:focus-visible,.helm-dashboard label:focus-visible{outline:3px solid #b15c00;outline-offset:3px}.helm-dashboard details[open]>summary{border-bottom:1px solid #d4dfef;margin-bottom:12px}.helm-dashboard pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f8;border-radius:8px;padding:14px;font-size:12px;max-height:480px;overflow:auto}
.helm-dashboard table{border-collapse:collapse;width:100%;font-size:13px;background:white;border-radius:10px}.helm-dashboard th{background:#e4edf9;color:#1b3c68;text-align:left}.helm-dashboard td,.helm-dashboard th{padding:10px 12px;border-bottom:1px solid #e2e9f2}/* Set every cell surface: notebook themes style odd/hover rows and pandas cells. */
.helm-dashboard table.dataframe{font-family:inherit;color:#243752;color-scheme:light}
.helm-dashboard table tbody tr,.helm-dashboard table tbody tr:nth-child(odd),.helm-dashboard table tbody tr:hover{background:#fff;color:#243752}
.helm-dashboard table tbody tr:nth-child(even){background:#f5f8fc;color:#243752}
.helm-dashboard table tbody td,.helm-dashboard table tbody th{background:inherit;color:#243752;font-family:inherit}
.helm-dashboard table thead th{background:#e4edf9;color:#1b3c68;font-family:inherit}
.helm-dashboard .table{overflow:auto}.helm-dashboard small{color:#4b617d}.helm-dashboard .helm-note{background:#edf4ff;border-left:4px solid #3067c0;padding:12px 16px;border-radius:6px}.helm-dashboard .helm-score{display:inline-flex;flex-direction:column;background:#e5f1ff;border:1px solid #bacfeb;border-radius:14px;padding:14px 22px;margin:12px 10px 12px 0;min-width:190px}.helm-dashboard .helm-score strong{font-size:30px;color:#163f80;line-height:1.2}.helm-dashboard .helm-score span{font-size:12px;text-transform:uppercase;letter-spacing:.7px;color:#385371}
.helm-dashboard .helm-chart{background:white;border:1px solid #d4dfef;border-radius:14px;padding:18px;margin:16px 0}.helm-dashboard .helm-chart figcaption{font-weight:750;color:#243f62;margin-bottom:12px}.helm-dashboard .helm-bar-row{display:grid;grid-template-columns:minmax(100px,1.3fr) minmax(80px,2fr) 95px;gap:12px;align-items:center;margin:9px 0}.helm-dashboard .helm-bar-label{overflow-wrap:anywhere;font-size:13px}.helm-dashboard .helm-track{height:14px;background:#e7edf5;border-radius:20px;overflow:hidden}.helm-dashboard .helm-fill{height:100%;border-radius:20px;min-width:0}.helm-dashboard .helm-positive{background:#b84b14}.helm-dashboard .helm-negative{background:#2367bb}.helm-dashboard .helm-number{font-size:13px;font-variant-numeric:tabular-nums;text-align:right}.helm-dashboard .helm-legend{font-size:12px;color:#4b617d}.helm-dashboard .helm-legend b:first-child{color:#a5410d}.helm-dashboard .helm-legend b:last-child{color:#1c58a1}.helm-dashboard .helm-chart ul{margin:8px 0}
@media(max-width:650px){.helm-dashboard .helm-panels{padding:12px}.helm-dashboard .helm-controls{padding:10px}.helm-dashboard .helm-header{padding:18px}.helm-dashboard .helm-header h1{font-size:23px}.helm-dashboard .helm-bar-row{grid-template-columns:1fr 1fr 78px;gap:6px}.helm-dashboard label{padding:8px 10px}.helm-dashboard section{padding:12px}}
"""


def contribution_chart(items, *, title, unit='coefficient', limit=8):
    pairs = [(str(name), float(value)) for name, value in items]
    if any(not math.isfinite(value) for _, value in pairs):
        raise ValueError('Les contributions affichées doivent être finies.')
    pairs = sorted(pairs, key=lambda x:abs(x[1]), reverse=True)[:limit]
    maximum = max((abs(v) for _, v in pairs), default=0) or 1
    body = '<figure class="helm-chart"><figcaption>' + escape(title) + '</figcaption>'
    body += '<p class="helm-legend"><b>Orange + : soutient la classe indiquée</b> · <b>Bleu − : s’y oppose</b>. Longueurs relatives au maximum affiché ; voir les valeurs signées.</p>'
    body += '<p class="helm-legend">' + ('Valeurs en points de probabilité.' if unit == 'probability' else 'Valeurs : coefficients locaux, non assimilables à des points de probabilité.') + '</p>'
    for name, value in pairs:
        display = f'{value*100:+.2f} pts' if unit == 'probability' else f'{value:+.4f}'
        color = 'helm-positive' if value >= 0 else 'helm-negative'
        body += f'<div class="helm-bar-row"><span class="helm-bar-label">{escape(name)}</span><div class="helm-track" aria-hidden="true"><div class="helm-fill {color}" style="width:{abs(value)/maximum*100:.4f}%"></div></div><span class="helm-number">{display}</span></div>'
    return body + '</figure>'


def score_card(label, value):
    return f'<div class="helm-score"><span>{escape(label)}</span><strong>{value:.2%}</strong></div>'


@dataclass
class Dashboard:
    """views[(group_key, profile_key)] contains trusted HELM-generated HTML."""
    title: str
    groups: dict
    profiles: dict
    views: dict
    default_profile: str
    subtitle: str = 'Choisissez un profil pour explorer les explications.'

    def to_html(self, path=None):
        if not self.groups or not self.profiles or self.default_profile not in self.profiles:
            raise ValueError('Groupes et profils non vides requis ; profil initial valide.')
        if set(self.views) != {(g,p) for g in self.groups for p in self.profiles}:
            raise ValueError('Une vue est requise pour chaque groupe et profil.')
        uid = 'helm-' + uuid4().hex
        radio = ''
        controls = ''
        selectors = []
        keys = {}
        for kind, options, default in [('profile',self.profiles,self.default_profile),('group',self.groups,next(iter(self.groups)))]:
            controls += '<fieldset><legend>' + ('Votre profil' if kind=='profile' else 'Dossiers ou commentaire') + '</legend>'
            for index,(key,label) in enumerate(options.items()):
                rid = f'{uid}-{kind}-{index}'
                keys[(kind,key)] = rid
                checked = ' checked' if key == default else ''
                radio += f'<input class="helm-radio" type="radio" name="{uid}-{kind}" id="{rid}" aria-label="{escape(label,quote=True)}"{checked}>'
                controls += f'<label for="{rid}">{escape(label)}</label>'
                selectors.append(f'#{rid}:checked~.helm-controls label[for="{rid}"]'+'{background:#215ac4;color:white;border-color:#215ac4}')
                selectors.append(f'#{rid}:focus-visible~.helm-controls label[for="{rid}"]'+'{outline:3px solid #b15c00;outline-offset:3px}')
            controls += '</fieldset>'
        panels = ''
        for index,((group,profile),html) in enumerate(self.views.items()):
            panel_id = f'{uid}-view-{index}'
            selectors.append(f'#{keys[("profile",profile)]}:checked~#{keys[("group",group)]}:checked~.helm-panels>#{panel_id}'+'{display:block}')
            panels += f'<article class="helm-view" id="{panel_id}">{html}</article>'
        result = '<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(self.title)+'</title><style>'+STYLE+''.join(selectors)+'</style><body><div class="helm-dashboard">'
        result += '<header class="helm-header"><div class="helm-brand">HELM / EXPLICABILITÉ</div><h1>'+escape(self.title)+'</h1><p>'+escape(self.subtitle)+'</p></header>'
        result += radio + '<div class="helm-controls">'+controls+'</div><main class="helm-panels">'+panels+'</main></div></body></html>'
        if path is not None:
            Path(path).write_text(result, encoding='utf-8')
        return result

    def _repr_html_(self):
        return self.to_html()
