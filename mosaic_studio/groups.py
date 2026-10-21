"""Editing operations on region groups (used by the app; each saves the project)."""
from __future__ import annotations

from .project import GROUP_DEFAULTS, new_group


def _renumber(project):
    for i, g in enumerate(project.ordered_groups()):
        g["order"] = i
    project.groups = [g for g in project.groups if g["members"]]
    for i, g in enumerate(project.ordered_groups()):
        g["order"] = i
    project.save()


def by_name(project, name):
    for g in project.groups:
        if g["name"] == name:
            return g
    return None


def drop_missing(project, lab):
    """Regions no longer in the label map leave their groups; empty groups go."""
    import numpy as np
    present = set(np.unique(lab).tolist())
    for g in project.groups:
        g["members"] = [m for m in g["members"] if m in present]
    project.groups = [g for g in project.groups if g["members"]]
    for i, g in enumerate(project.ordered_groups()):
        g["order"] = i


def fuse(project, regions):
    """Fuse regions into one region (the lowest number): the borders between them
    disappear, so the mosaic treats them as a single blob. Returns the new id."""
    regions = sorted(set(int(r) for r in regions))
    if len(regions) < 2:
        return regions[0] if regions else None
    import numpy as np
    lab = project.regions()
    keep = regions[0]
    lab[np.isin(lab, regions[1:])] = keep
    gone = set(regions[1:])
    for g in project.groups:
        g["members"] = [m for m in g["members"] if m not in gone]
    project.set_regions(lab)
    return keep


def add_to_group(project, regions, name):
    """Merge regions into the named group: they are taken out of their old groups
    and fused with the group's regions into one blob (no borders left inside)."""
    target = by_name(project, name)
    if target is None:
        return
    regions = set(int(r) for r in regions)
    for g in project.groups:
        if g is not target:
            g["members"] = [m for m in g["members"] if m not in regions]
    target["members"] = sorted(set(target["members"]) | regions)
    target["members"] = [fuse(project, target["members"])]
    _renumber(project)


def new_group_from(project, regions, name, kind="object"):
    """New group (at the front) made of the regions fused into one; returns its
    name, which gets a number added if the name is already taken."""
    regions = set(int(r) for r in regions)
    if not regions:
        return None
    for g in project.groups:
        g["members"] = [m for m in g["members"] if m not in regions]
    base, k = name.strip() or "group", 2
    while by_name(project, name):
        name = f"{base} {k}"; k += 1
    gid = max([g["id"] for g in project.groups] + [0]) + 1
    merged = fuse(project, regions)                                   # one blob
    project.groups.append(new_group(gid, name, kind, [merged], -0.5))  # goes to the front
    _renumber(project)
    return name


def rename(project, old, new):
    g = by_name(project, old)
    if g and new.strip() and not by_name(project, new.strip()):
        g["name"] = new.strip()
        project.save()


def set_kind(project, name, kind):
    """Switch object/background; also resets its style to that kind's default."""
    g = by_name(project, name)
    if g and g["kind"] != kind:
        g["kind"] = kind
        g.update(dict(GROUP_DEFAULTS[kind]))
        project.save()


def move(project, name, delta):
    """delta = -1: one step to the front, +1: one step to the back."""
    order = project.ordered_groups()
    g = by_name(project, name)
    if g is None:
        return
    i = order.index(g)
    j = min(max(i + delta, 0), len(order) - 1)
    order.insert(j, order.pop(i))
    for k, h in enumerate(order):
        h["order"] = k
    project.save()


def set_style(project, name, **kw):
    g = by_name(project, name)
    if g:
        g.update(kw)
        project.save()


def table(project):
    rows = []
    for g in project.ordered_groups():
        extra = f"{g['angle']:.0f}°" if g["style"] in ("brick", "grid", "rows") else ""
        rows.append([g["order"] + 1, g["name"], g["kind"], f"{g['style']} {extra}".strip(),
                     g["outline_rows"], g["colors"], ", ".join(map(str, g["members"]))])
    return rows
