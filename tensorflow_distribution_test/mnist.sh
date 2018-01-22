#! /bin/bash
p2p ()
{
    local task_num=$2
    local job_=$1
    python test_ps_mnist.py --job_name=${job_} --ps_hosts='172.20.1.34:2428' --worker_hosts='172.20.1.35:2428,172.20.1.36:2428' --task_index=${task_num}
}

p3p ()
{
    local task_num=$2
    local job_=$1
    python test_ps_mnist.py --job_name=${job_} --ps_hosts='172.20.1.34:2428' --worker_hosts='172.20.1.34:2427,172.20.1.35:2428,172.20.1.36:2428' --task_index=${task_num}
}

case "${1}" in
    "p2p")
        echo "p2p will be executed"
        p2p $2 $3
        ;;
    "p3p")
        echo "p3p will be executed"
        p3p $2 $3
        ;;
esac;
