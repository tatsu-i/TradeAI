DOCKER_HUB_USER="tatsui"
for image in $(ls docker)
do
	CWD=$(pwd)
	cd docker/$image
	if [ -e version ]; then
		VERSION=$(cat version)
		IMAGE_NAME=$DOCKER_HUB_USER/$image:$VERSION
		echo "Build image $IMAGE_NAME"
		docker build -t $IMAGE_NAME .
		docker push $IMAGE_NAME
	fi
	cd $CWD
done
