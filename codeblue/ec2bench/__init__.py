"""Run the Code Blue study on EC2-Bench (github.com/OpenBMB/EC2-Bench, arXiv 2610.17816).

  labels.py   EC2-Bench's sensitive values (gt/sensitive_items.json) -> our secret labels, plus our hand labels
              (essential secrets, attacker-writable files, composite secrets) from labels/ec2bench.json
  p3_view.py  P3 for EC2-Bench's cloud proxy. Copied into their tree by patch.py; stdlib only
  patch.py    patch an EC2-Bench checkout so delegation mode can send P3 instead of the full history (P0)
  inject.py   copy an EC2-Bench data root with the stuck_and_stuff injection planted in the labelled files
  collect.py  read EC2-Bench run folders and produce our claim.json, rescue curves and mosaic tables
"""
