from lib_es.extra_schedulers.log_linear import log_linear_scheduler

import modules.sd_schedulers as sched

extra_scheduler_list = [
    sched.Scheduler("linear_log", "Linear Log", log_linear_scheduler, need_inner_model=True),
]


def add_schedulers():
    """
    Add extra schedulers to the list of schedulers in the webui.
    """
    for scheduler in extra_scheduler_list:
        if scheduler.name not in sched.schedulers_map:
            sched.schedulers.append(scheduler)
            sched.schedulers_map = {**{x.name: x for x in sched.schedulers}, **{x.label: x for x in sched.schedulers}}
