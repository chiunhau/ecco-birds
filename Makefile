REMOTE      := chiunhau@puhti.csc.fi
REMOTE_PATH := /scratch/project_2017429/chiunhau/birds
RSYNC       := rsync -avz --progress \
               --exclude='.DS_Store' \
               --exclude='__pycache__' \
               --exclude='*.pyc'

RSYNC_PULL  := $(RSYNC) --exclude='books/'

.PHONY: push pull ssh

## push local works/ → puhti
push:
	$(RSYNC) works/ $(REMOTE):$(REMOTE_PATH)/works/

## pull puhti works/ → local  (excludes books/)
pull:
	$(RSYNC_PULL) $(REMOTE):$(REMOTE_PATH)/works/ works/

## open ssh session at the remote project dir
ssh:
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && source /scratch/project_2017429/chiunhau/my_python_env/bin/activate && exec $$SHELL"