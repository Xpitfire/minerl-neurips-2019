
base:
	docker build -f base.dockerfile -t ml-jku/base .

vnc: base
	docker build -f vnc.dockerfile -t ml-jku/vnc .

minerl-gpu: vnc
	docker build -f minerl-gpu.dockerfile -t ml-jku/minerl-gpu .


.phony: base vnc minerl-gpu