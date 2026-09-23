"""Catenary receptivity policy for the DoE -- one switch, read everywhere.

``RECEPTIVE`` decides whether a train can return braking energy to the wire.
Every substation this study prices is receptive hardware: the 25 kV AC feeder
stations are transformers, bidirectional by construction, and the 9 kV MVDC
stations are costed explicitly as VSC converters whose "bidirectional
regeneration-to-grid" is the stated rationale for their price premium (see
``ARCH_MEDIUM_COSTS`` in the catenary app). Denying the return while paying for
the equipment was an inconsistency; ``True`` removes it. Set ``False`` to
recover the pre-2026-09 behaviour (all braking beyond the battery goes to the
dynamic-brake grids) -- the flag reaches the Rust ``CatenaryInterface`` through
``apply_to_locomotive`` and is part of the sweep's input fingerprint, so
flipping it invalidates the catenary cells and nothing else.

``EXPORT_CREDIT_FRACTION`` is what returned energy is worth, as a fraction of
the purchase price. Energy that offsets the railroad's own coincident draw is
worth the full price; net export to the utility is usually credited at an
avoided-cost or wholesale rate well below retail, and only with an
interconnection agreement. The simulation cannot separate the two (it does
not model other trains' coincident draw at the same substation), so one blended
factor stands in for both. Energy and emissions accounting use the full net
flow regardless -- a returned kWh displaces a generated kWh whoever is paid.
"""

RECEPTIVE = True
EXPORT_CREDIT_FRACTION = 0.5


def apply_to_locomotive(loco):
    """Stamp ``RECEPTIVE`` on a locomotive's catenary interface, if it has one.

    Pydict round trip, like every other locomotive edit in this repo; a
    locomotive without a pantograph is returned unchanged.
    """
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    body = d["loco_type"][kind] or {}
    cat = body.get("cat_int")
    if not isinstance(cat, dict):
        return loco
    if bool(cat.get("receptive", False)) == RECEPTIVE:
        return loco
    cat["receptive"] = RECEPTIVE
    import altrios as alt
    return alt.Locomotive.from_pydict(d)


def split_line_loss(draw_j: float, regen_j: float, line_loss_j: float):
    """Apportion the (direction-blind) resistive line loss between the draw and
    the return by energy share. Returns ``(loss_on_draw, loss_on_return)``."""
    total = draw_j + regen_j
    if total <= 0.0:
        return line_loss_j, 0.0
    on_return = line_loss_j * regen_j / total
    return line_loss_j - on_return, on_return


def grid_flows(draw_j: float, regen_j: float, line_loss_j: float):
    """``(grid_draw_j, grid_return_j)``: what the substation supplies (pantograph
    draw plus its line loss) and what it receives (pantograph return minus the
    line loss on the way back, floored at zero)."""
    loss_draw, loss_return = split_line_loss(draw_j, regen_j, line_loss_j)
    return draw_j + loss_draw, max(regen_j - loss_return, 0.0)
