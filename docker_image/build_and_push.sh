export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build -t duesenfranz/gh_action_poetry .
docker tag duesenfranz/gh_action_poetry ghcr.io/duesenfranz/gh_action_poetry
docker push ghcr.io/duesenfranz/gh_action_poetry
