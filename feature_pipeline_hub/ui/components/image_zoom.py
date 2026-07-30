"""Shared "click the thumbnail to inspect it" behaviour.

Used from every grid that shows thumbnails (curation, quality checks): the
thumbnail answers "which image is this", the modal answers "is this image good
enough to train on".

Streamlit has no click event for `st.image`, so the click target is a real button
stretched over the thumbnail and made invisible in CSS. That keeps the interaction
where the user expects it — on the image — without a custom JS component, and the
button still carries the keyboard focus and tooltip a plain <img> would not.
"""

import streamlit as st

import state

# Keyed containers and widgets get a `st-key-<key>` class, and the keys below are
# per-sample, so these rules match on the prefix instead of naming each one. The
# two prefixes are deliberately dissimilar ("frame"/"hit"): a shared one would make
# the substring selectors match each other's elements.
_STYLES = """
<style>
[class*="st-key-zoomframe_"] {
    position: relative;
}
/* `inset: 0` alone leaves the hit area the width of the empty button, so pin the
   size explicitly. Streamlit's own rules set both, hence `!important` throughout. */
[class*="st-key-zoomhit_"] {
    position: absolute;
    inset: 0;
    width: 100% !important;
    height: 100% !important;
    z-index: 1;
}
[class*="st-key-zoomhit_"] > div {
    width: 100% !important;
    height: 100% !important;
}
/* Stretching the button by inheritance does not survive the wrapper chain: a
   `height: 100%` resolved against an auto-height wrapper collapses, leaving a
   1px-tall strip that only catches clicks along the very top edge. The hit box is
   already positioned, so size the button against that instead of its parent. */
[class*="st-key-zoomhit_"] button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
}
[class*="st-key-zoomhit_"] button,
[class*="st-key-zoomhit_"] button:hover,
[class*="st-key-zoomhit_"] button:focus,
[class*="st-key-zoomhit_"] button:active {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
    cursor: zoom-in;
}
/* The button is invisible, so the hover affordance has to come from the image. */
[class*="st-key-zoomframe_"]:has([class*="st-key-zoomhit_"] button:hover) img {
    filter: brightness(1.1);
}
</style>
"""


def inject_styles() -> None:
    """Emit the stylesheet the click targets need. Call once per page render."""
    st.html(_STYLES)


@st.dialog(" ", width="large")
def render_dialog(image_path: str) -> None:
    """Full-size inspection view: the original file, never a re-encoded copy.

    This is where the user judges focus and compression artefacts, so it goes
    through `render_original_image` rather than `st.image` directly — the latter
    would quietly resample and recompress anything over 1460px wide.

    The toggle is a widget inside a dialog, so it reruns only this function rather
    than the whole page, and the dialog stays open across the switch.
    """
    actual_size = st.toggle(
        "Tamaño real (100 %)",
        key="zoom_actual_size",
        help="Ver los píxeles sin escalar, con scroll, en lugar de ajustar la imagen al diálogo",
    )
    state.render_original_image(image_path, actual_size=actual_size)


def clickable_thumbnail(image_path: str, key: str, **thumbnail_kwargs) -> None:
    """A `state.render_thumbnail` that opens the full-size modal when clicked.

    `thumbnail_kwargs` are passed straight through, so callers swap one call for
    the other and keep their sizing.
    """
    # A pixel-width thumbnail does not fill its column, and the overlay covers the
    # frame rather than the image — so hug the image, or the empty space beside it
    # would be clickable too. Stretched thumbnails already span the frame.
    hugs_image = isinstance(thumbnail_kwargs.get("width"), int)

    with st.container(key=f"zoomframe_{key}", width="content" if hugs_image else "stretch"):
        state.render_thumbnail(image_path, **thumbnail_kwargs)
        if st.button("", key=f"zoomhit_{key}", help="Ver en tamaño grande"):
            render_dialog(image_path)
