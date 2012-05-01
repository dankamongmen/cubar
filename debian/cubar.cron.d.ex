#
# Regular cron jobs for the cubar package
#
0 4	* * *	root	[ -x /usr/bin/cubar_maintenance ] && /usr/bin/cubar_maintenance
