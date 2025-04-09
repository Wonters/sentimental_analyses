ovhai job run \
	--name ai-training-pytorch-tweet \
	--flavor l4-1-gpu \
	--ssh-public-keys "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCyAD1eL6QQn+mW1GzmmA9CKWzH28IjBkXJqpJqa+EU2i7AqHQMNVEEXNssQanIW85ptBBsb46C0UFsmmZIXwyZUqQAfPbh7KdLM7IT+D78i6SzbQs0qyAGUpIjp9TmaJ3vwhuJygKVSv8DtCNS+egWhQkyK3KYJCRwG3su9f2Cw5S5fJ3tab6mhU4z53LJhRe+oQ4M34HtSEBK6HQSojHID9Lhe5OTwo+T8IgztGNdDDMtSbrZJg2GYvjj65PslVGNfjX63hW2rNeOZwGnpu9eNURXBlnnkKXGP5aUhTK1y0a1stih+F+mZIInNj7qdDu9S7onxe0enmgYjq7GXWFwKceOTEhK+qnzaeehTKF199ijylelCfJ1Kid2E29/1zxxNMuXiczSmJZEpMbhvW/J/HTlBteVsvoBb+Crg5o0ZYkSM6C5IJM9k01LQ4VlsWu5EJft2wOroCUJUU7De/wf/WVS26Vh7AVJAhOSwlDPstxV3H+TIf5CAFpG/4Z7HnM= wonters@Mac-Pro-de-wonters.local" \
	--unsecure-http ovhcom/ai-training-pytorch:2.4.0 \
	--volume datastore-model@GRA/:/workspace/data:RW \
	-- bash -c '
	git clone --branch develop https://github.com/Wonters/sentimental_analyses.git /workspace/app &&
	cd /workspace/app &&
	./train.sh
	'