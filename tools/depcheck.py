import importlib.metadata as m
for pkg in ['brax', 'playground', 'mujoco', 'mujoco-mjx', 'jax', 'jaxlib',
            'jax-cuda12-plugin', 'jax-cuda12-pjrt']:
    try:
        d = m.distribution(pkg)
        reqs = [r for r in (d.requires or []) if 'jax' in r.lower() or 'cuda' in r.lower()]
        print(f"{pkg} == {d.version}")
        for r in reqs:
            print("    ", r)
    except Exception as e:
        print(pkg, "ERR", e)
