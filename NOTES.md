# Overview 

## v1

### Config files

* `configs-base.yml`: base experiments (cell nuclei)
* `configs-edge.yml`: edge experiments (cell border)

### Data prep 

`prep/prepare.py`

* `create_raw(...)` ==> `/data/raw/flame/proc/raw`

NOTE: these experiments failed due to poor annotations

## v2

* `configs-v00.yml`: cell nuclei + border (v00 dataset)
* `configs-v01.yml`: cell nuclei + border (v01 dataset)

### Data prep 

`prep/prepare.py`

* `create_v00(...)` ==> `/data/raw/flame/proc/v00`
* `create_v01(...)` ==> `/data/raw/flame/proc/v01` (NEWEST DATASET)
