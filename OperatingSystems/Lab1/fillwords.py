info = "17341015CHZ"
infob = bytes(info,"ascii")
myname = bytearray(open("formatted.img","rb").read())
output = open("myname.img","wb")
j = 0
# for i in range(513):
for i in range(myname.find(b"\xfe")+1,510):
	myname[i] = ord(info[j])
	j = (j+1) % len(info)
output.write(myname)