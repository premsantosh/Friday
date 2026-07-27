"""
Form harvesting — read any page's form as structured data, with no site knowledge.

One `page.evaluate()` walk collects every fillable control together with the
label a human would read next to it, resolved the way a screen reader does:
`el.labels` first (which handles both `<label for>` and wrapping labels), then
ARIA, then the enclosing field container's text, then the placeholder. Nothing
here knows what OpenTable or a clinic intake form looks like.

Two properties the rest of the package depends on:

  - **Stable refs.** A field's ref is derived from `name`, else `id`, else its
    label, else its ordinal. `locator_for()` turns a ref back into a Playwright
    locator on a freshly loaded page, so a plan approved in `prepare()` can be
    replayed in `commit()` minutes later.
  - **Refusal beats improvisation.** Password and payment-card inputs make the
    whole snapshot blocked: card handling belongs to the Privacy.com path in
    payment.py, never to a generic form filler. File uploads are skipped and
    reported rather than guessed at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Filling these would mean handling credentials or card data through a path
# with none of the protections payment.py has.
BLOCK_PASSWORD = "password_field"
BLOCK_CARD = "payment_card_field"


@dataclass
class FormOption:
    label: str
    value: str

    def to_dict(self) -> Dict[str, str]:
        return {"label": self.label, "value": self.value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormOption":
        return cls(label=d.get("label", ""), value=d.get("value", ""))


@dataclass
class FormField:
    ref: str                       # stable handle; survives a page reload
    tag: str                       # input | select | textarea
    type: str                      # text | email | tel | date | checkbox | radio | select | ...
    label: str = ""
    name: str = ""
    element_id: str = ""
    placeholder: str = ""
    required: bool = False
    options: List[FormOption] = field(default_factory=list)
    maxlength: Optional[int] = None
    ordinal: int = 0               # position among same-tag controls, last-resort locator

    @property
    def describes_choice(self) -> bool:
        return self.type in ("select", "radio")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref, "tag": self.tag, "type": self.type, "label": self.label,
            "name": self.name, "element_id": self.element_id,
            "placeholder": self.placeholder, "required": self.required,
            "options": [o.to_dict() for o in self.options],
            "maxlength": self.maxlength, "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormField":
        return cls(
            ref=d["ref"], tag=d.get("tag", "input"), type=d.get("type", "text"),
            label=d.get("label", ""), name=d.get("name", ""),
            element_id=d.get("element_id", ""), placeholder=d.get("placeholder", ""),
            required=bool(d.get("required")),
            options=[FormOption.from_dict(o) for o in d.get("options") or []],
            maxlength=d.get("maxlength"), ordinal=int(d.get("ordinal") or 0),
        )

    def describe(self) -> Dict[str, Any]:
        """The projection sent to the mapper: page content only, no user data."""
        out: Dict[str, Any] = {
            "ref": self.ref, "type": self.type,
            "label": self.label or self.placeholder or self.name,
            "required": self.required,
        }
        if self.placeholder and self.placeholder != out["label"]:
            out["placeholder"] = self.placeholder
        if self.options:
            out["options"] = [o.label for o in self.options]
        if self.maxlength:
            out["maxlength"] = self.maxlength
        return out


@dataclass
class FormSnapshot:
    url: str = ""
    heading: str = ""
    fields: List[FormField] = field(default_factory=list)
    submit_labels: List[str] = field(default_factory=list)
    form_count: int = 0
    has_file_field: bool = False
    blocked_reason: Optional[str] = None    # set → refuse to touch this form
    # A bot-detection widget guards submission. We can still fill the form, but
    # pressing submit ourselves is pointless: reCAPTCHA v3 scores the whole
    # session, so a programmatically-filled form is rejected even when a human
    # clicks the button (measured against backontrack-pt.com, 2026-07-26).
    # Raising that score means defeating the control, which the spec forbids.
    human_submit_required: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.blocked_reason is None and bool(self.fields)

    def by_ref(self, ref: str) -> Optional[FormField]:
        return next((f for f in self.fields if f.ref == ref), None)

    def required_refs(self) -> List[str]:
        return [f.ref for f in self.fields if f.required]

    def describe(self) -> Dict[str, Any]:
        return {"url": self.url, "heading": self.heading,
                "fields": [f.describe() for f in self.fields]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url, "heading": self.heading,
            "fields": [f.to_dict() for f in self.fields],
            "submit_labels": self.submit_labels, "form_count": self.form_count,
            "has_file_field": self.has_file_field, "blocked_reason": self.blocked_reason,
            "human_submit_required": self.human_submit_required,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormSnapshot":
        return cls(
            url=d.get("url", ""), heading=d.get("heading", ""),
            fields=[FormField.from_dict(f) for f in d.get("fields") or []],
            submit_labels=list(d.get("submit_labels") or []),
            form_count=int(d.get("form_count") or 0),
            has_file_field=bool(d.get("has_file_field")),
            blocked_reason=d.get("blocked_reason"),
            human_submit_required=d.get("human_submit_required"),
        )


# --------------------------------------------------------------------- the DOM walk
#
# Runs in the page. Returns plain JSON. Deliberately structural: it keys off
# HTML semantics (labels, ARIA, input types) that every form shares, never off
# a particular site's class names.
_HARVEST_JS = r"""
() => {
  const SKIP_TYPES = new Set(['hidden', 'submit', 'button', 'reset', 'image']);
  const CARD_HINT = /(card[_-]?number|cardnum|cc[_-]?num|creditcard|cvv|cvc|security[_-]?code|exp(iry|iration)?[_-]?(date|month|year)?)/i;
  const HONEYPOT = /(honeypot|honey[_-]?pot|\bhp[_-]|[_-]hp\b|bot[_-]?field|leave[_-]?(this[_-]?)?blank|do[_-]?not[_-]?fill|url_check|comment_field_x)/i;

  const text = (el) => (el ? (el.textContent || '').replace(/\s+/g, ' ').trim() : '');

  const visible = (el) => {
    const st = window.getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    if (parseFloat(st.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    // Off-screen positioning is the classic honeypot trick.
    if (r.width <= 1 || r.height <= 1) return false;
    if (r.left < -1000 || r.top < -5000) return false;
    return true;
  };

  const isHoneypot = (el) => {
    if (el.closest('[aria-hidden="true"]')) return true;
    const id = [el.name, el.id, el.className, el.getAttribute('autocomplete') || ''].join(' ');
    if (HONEYPOT.test(id)) return true;
    let node = el.parentElement, depth = 0;
    while (node && depth++ < 3) {
      if (!visible(node)) return true;
      node = node.parentElement;
    }
    return false;
  };

  const labelFor = (el) => {
    // 1. Native association: <label for>, and labels wrapping the control.
    if (el.labels && el.labels.length) {
      for (const l of el.labels) { const t = text(l); if (t) return t; }
    }
    // 2. ARIA.
    const aria = (el.getAttribute('aria-label') || '').trim();
    if (aria) return aria;
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const parts = by.split(/\s+/).map(id => text(document.getElementById(id))).filter(Boolean);
      if (parts.length) return parts.join(' ');
    }
    // 3. A legend, for a control inside a fieldset of its own.
    const fs = el.closest('fieldset');
    if (fs) { const lg = text(fs.querySelector('legend')); if (lg && fs.querySelectorAll('input,select,textarea').length <= 6) return lg; }
    // 4. The enclosing field container's own text, minus any nested controls'.
    let node = el.parentElement, depth = 0;
    while (node && depth++ < 4) {
      const clone = node.cloneNode(true);
      clone.querySelectorAll('input,select,textarea,button,option,script,style').forEach(n => n.remove());
      const t = text(clone);
      if (t && t.length <= 160) return t;
      node = node.parentElement;
    }
    // 5. Last resort.
    return (el.placeholder || '').trim();
  };

  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);

  const collect = (root) => {
    const out = { fields: [], blocked: null, hasFile: false };
    const seenRadio = new Set();
    const els = Array.from(root.querySelectorAll('input, select, textarea'));
    const ordinals = {};
    for (const el of els) {
      const tag = el.tagName.toLowerCase();
      const type = tag === 'select' ? 'select'
                 : tag === 'textarea' ? 'textarea'
                 : (el.getAttribute('type') || 'text').toLowerCase();
      if (SKIP_TYPES.has(type)) continue;

      if (type === 'password') { out.blocked = 'password_field'; continue; }
      const auto = (el.getAttribute('autocomplete') || '').toLowerCase();
      const ident = [el.name || '', el.id || '', el.getAttribute('data-name') || ''].join(' ');
      if (auto.startsWith('cc-') || CARD_HINT.test(ident)) { out.blocked = 'payment_card_field'; continue; }
      if (type === 'file') { out.hasFile = true; continue; }

      if (!visible(el) || isHoneypot(el)) continue;

      const name = el.name || '';
      if (type === 'radio') {
        const groupKey = name || ('radio@' + slug(labelFor(el)));
        if (seenRadio.has(groupKey)) continue;
        seenRadio.add(groupKey);
        const group = name
          ? Array.from(root.querySelectorAll(`input[type=radio][name="${CSS.escape(name)}"]`))
          : [el];
        const options = group.map(r => ({ label: labelFor(r) || r.value, value: r.value }));
        // The group's own question, not the first option's label.
        const fs2 = el.closest('fieldset');
        const question = (fs2 && text(fs2.querySelector('legend')))
          || el.getAttribute('aria-label')
          || (el.closest('[role=radiogroup]') && el.closest('[role=radiogroup]').getAttribute('aria-label'))
          || name;
        ordinals[tag] = (ordinals[tag] || 0);
        out.fields.push({
          ref: '', tag, type: 'radio', label: question, name, element_id: el.id || '',
          placeholder: '', required: group.some(r => r.required), options,
          maxlength: null, ordinal: ordinals[tag]++,
        });
        continue;
      }

      const options = tag === 'select'
        ? Array.from(el.options)
            .filter(o => o.value !== '' || (o.textContent || '').trim() !== '')
            .map(o => ({ label: (o.textContent || '').replace(/\s+/g, ' ').trim(), value: o.value }))
        : [];

      const label = labelFor(el);
      ordinals[tag] = (ordinals[tag] || 0);
      out.fields.push({
        ref: '', tag, type, label, name, element_id: el.id || '',
        placeholder: (el.placeholder || '').trim(),
        required: !!el.required || el.getAttribute('aria-required') === 'true'
                  || /\*\s*$/.test(label),
        options,
        maxlength: el.maxLength && el.maxLength > 0 ? el.maxLength : null,
        ordinal: ordinals[tag]++,
      });
    }
    return out;
  };

  const submitLabels = (root) => Array.from(
      root.querySelectorAll('button, input[type=submit], [role=button]'))
    .filter(b => visible(b))
    .map(b => (b.value || b.textContent || b.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim())
    .filter(Boolean).slice(0, 8);

  // Score forms structurally so a header search box never beats the real form.
  const score = (c) => {
    const types = c.fields.map(f => f.type);
    if (c.fields.length === 0) return -100;
    if (c.fields.length === 1 && /search|query|^q$/i.test(c.fields[0].name + ' ' + c.fields[0].label)) return -50;
    let s = c.fields.length;
    if (types.includes('email')) s += 3;
    if (types.includes('textarea')) s += 2;
    if (types.includes('tel')) s += 2;
    return s;
  };

  const forms = Array.from(document.querySelectorAll('form'));
  const roots = forms.length ? forms : [document.body];
  let best = null, bestRoot = null, bestScore = -Infinity, blocked = null, hasFile = false;
  for (const root of roots) {
    const c = collect(root);
    if (c.blocked) blocked = c.blocked;
    if (c.hasFile) hasFile = true;
    const s = score(c);
    if (s > bestScore) { bestScore = s; best = c; bestRoot = root; }
  }
  if (!best) best = { fields: [], blocked: null, hasFile: false };

  // Unique, stable refs: name > id > label slug > tag:type:ordinal.
  const used = new Set();
  for (const f of best.fields) {
    let base = f.name || f.element_id || slug(f.label) || `${f.tag}_${f.type}_${f.ordinal}`;
    let ref = base, n = 2;
    while (used.has(ref)) ref = `${base}__${n++}`;
    used.add(ref);
    f.ref = ref;
  }

  // Bot-detection widgets that gate submission. Detected by their own markup
  // and script tags — we only look, never touch.
  const captcha = document.querySelector(
      '.g-recaptcha, .elementor-g-recaptcha, [class*="recaptcha"], [data-sitekey],'
    + ' .h-captcha, .cf-turnstile, iframe[src*="recaptcha"], iframe[src*="hcaptcha"]')
    || Array.from(document.querySelectorAll('script[src]')).find(
         s => /recaptcha|hcaptcha|turnstile/i.test(s.src));
  let humanSubmit = null;
  if (captcha) {
    const blob = (captcha.className || '') + ' ' + (captcha.src || '') + ' '
               + (captcha.getAttribute ? (captcha.getAttribute('data-type') || '') : '');
    humanSubmit = /hcaptcha/i.test(blob) ? 'hcaptcha'
                : /turnstile/i.test(blob) ? 'turnstile'
                : 'recaptcha';
  }

  const heading = (document.querySelector('h1, h2') || {}).textContent || document.title || '';
  return {
    url: location.href,
    heading: heading.replace(/\s+/g, ' ').trim().slice(0, 200),
    fields: best.fields,
    submit_labels: submitLabels(bestRoot === document.body ? document : bestRoot),
    form_count: forms.length,
    has_file_field: hasFile,
    blocked_reason: best.blocked || blocked,
    human_submit_required: humanSubmit,
  };
}
"""


async def harvest(page) -> FormSnapshot:
    """Read the most form-like region of the current page."""
    raw = await page.evaluate(_HARVEST_JS)
    snapshot = FormSnapshot.from_dict(raw)
    if snapshot.blocked_reason:
        logger.info("Refusing form at %s: %s", snapshot.url, snapshot.blocked_reason)
    return snapshot


# ------------------------------------------------------------------ re-resolution

_TAG_SELECTOR = {"input": "input", "select": "select", "textarea": "textarea"}


async def _first_visible(locator):
    """The first *visible* match, or None.

    Responsive pages routinely render the same form twice (mobile + desktop) and
    hide one; a bare `.first` picks the hidden copy about half the time. The
    OpenTable channel already learned this the hard way.
    """
    try:
        count = await locator.count()
    except Exception:
        return None
    if count == 0:
        return None
    for i in range(min(count, 8)):
        candidate = locator.nth(i)
        try:
            if await candidate.is_visible():
                return candidate
        except Exception:
            continue
    return None


async def locator_for(page, field: FormField, option_value: Optional[str] = None):
    """Re-resolve a harvested field on a freshly loaded page.

    Returns a Playwright locator or None. None means the page changed enough
    that we can't be sure which control this is — the caller hands off rather
    than typing into the wrong box.
    """
    tag = _TAG_SELECTOR.get(field.tag, "input")

    if field.name:
        selector = f'{tag}[name="{_css_quote(field.name)}"]'
        if field.type == "radio" and option_value is not None:
            selector = (f'input[type="radio"][name="{_css_quote(field.name)}"]'
                        f'[value="{_css_quote(option_value)}"]')
        found = await _first_visible(page.locator(selector))
        if found is not None:
            return found

    if field.element_id:
        found = await _first_visible(page.locator(f'[id="{_css_quote(field.element_id)}"]'))
        if found is not None:
            return found

    if field.label:
        for exact in (True, False):
            try:
                found = await _first_visible(page.get_by_label(field.label, exact=exact))
            except Exception:
                found = None
            if found is not None:
                return found

    if field.placeholder:
        try:
            found = await _first_visible(page.get_by_placeholder(field.placeholder))
        except Exception:
            found = None
        if found is not None:
            return found

    # Ordinal is a genuine last resort: right only if the form's shape is intact.
    try:
        by_ordinal = page.locator(tag).nth(field.ordinal)
        if await by_ordinal.count() > 0 and await by_ordinal.is_visible():
            return by_ordinal
    except Exception:
        pass
    return None


def _css_quote(value: str) -> str:
    """Escape a value for use inside a double-quoted CSS attribute selector."""
    return re.sub(r'(["\\])', r"\\\1", value or "")
