sleep 10

name=custom10

for i in {0..10}
do
echo $i
gphoto2 --capture-image-and-download --filename $name/img$i.%C
sleep 5
done