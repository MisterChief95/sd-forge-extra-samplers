import lib_es.extra_schedulers

import modules.sd_schedulers as sched


extra_scheduler_list = [
    sched.Scheduler("linear_log", "Linear Log", lib_es.extra_schedulers.log_linear_scheduler, need_inner_model=True),
]

for scheduler in extra_scheduler_list:
    sched.schedulers.append(scheduler)
    sched.schedulers_map = {**{x.name: x for x in sched.schedulers}, **{x.label: x for x in sched.schedulers}}
