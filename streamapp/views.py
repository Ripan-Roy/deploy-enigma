from django.shortcuts import render
from django.http.response import StreamingHttpResponse,HttpResponse
from streamapp.camera import ThermalDetect, VideoCamera, IPWebCam, MaskDetect, LiveWebCam
# Create your views here.


def index(request):
	cam_type = request.GET.get('cam_type',None)
	is_thermal=False
	if cam_type == "" or cam_type == None:
		is_thermal = False
	else:
		is_thermal=True
	context={
		'is_thermal':is_thermal
	}
	return render(request, 'streamapp/home.html',context)


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def webcam_feed(request):
	return StreamingHttpResponse(gen(IPWebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def mask_feed(request):
	return StreamingHttpResponse(gen(MaskDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')
					
def livecam_feed(request):
	return StreamingHttpResponse(gen(LiveWebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')
def ripandabody(request):
	return StreamingHttpResponse(gen(ThermalDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')