from lib_es.extra_schedulers import __all_schedulers__

import modules.sd_schedulers as sched


extra_scheduler_list = [
    sched.Scheduler(fn.name, fn.alias, fn, need_inner_model=fn.need_inner_model) for fn in __all_schedulers__
]


def add_schedulers():
    """
    Add extra schedulers to the list of schedulers in the webui.
    """
    for scheduler in extra_scheduler_list:
        if scheduler.name not in sched.schedulers_map:
            sched.schedulers.append(scheduler)
            sched.schedulers_map = {**{x.name: x for x in sched.schedulers}, **{x.label: x for x in sched.schedulers}}
