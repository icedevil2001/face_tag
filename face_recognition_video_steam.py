#!/usr/bin/env python 

import cv2
import face_recognition
from pathlib import Path
import numpy as np
# from imutils.video import VideoStream
# import imutils
import time 
import datetime 
from collections import defaultdict

from sqlalchemy import (
		create_engine,
		Column, Integer,
		String, ForeignKey,
		DateTime, Date, Time, Float 
		)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship


Base = declarative_base()

class Attendance(Base):
	__tablename__ = 'person'

	id = Column('id', Integer, primary_key=True)
	## https://stackoverflow.com/questions/13370317/sqlalchemy-default-datetime
	# date = Column('date', Date)  ## default=datetime.datetime.utcnow 
	# time = Column('time', Time)
	date = Column('date', String) 
	time = Column('time', String)  
	attendies = Column('attendies', String)
	distances = Column('distance_mean', Float) # euclidean distance < 0.6 for a good match 
	observation = Column('observations', Integer)
	sites = Column('sites', String)

	def __repr__(self):
		return f'{self.id} {self.date} {self.time} {self.attendies} {self.distances} {self.observation} {self.sites}'
        



engine = create_engine('sqlite:///database/Attendance.db', echo=True)
Base.metadata.create_all(bind=engine)  ## create the table if present 

Session = sessionmaker(bind=engine)
session = Session()



def add_attendee(data, site, echo=False):
	## https://stackoverflow.com/questions/32938475/flask-sqlalchemy-check-if-row-exists-in-table

	# exists = db.session.query(User.id).filter_by(name='davidism').scalar() is not None
	
	if len(data.keys())==0: 
		print('No data', len(data), data)
		return None
	for name, dist in data.items():
		observation = len(dist) 
		if observation< 5:
			print('observation:', observation)
			continue
		dt = datetime.datetime.now()
		date = dt.strftime('%Y/%m/%d')
		time = dt.strftime('%H:%M:%S')
		# mean_dist = np.mean(dist)
		attendance = Attendance()  
		attendance.date = date
		attendance.time = time
		attendance.attendies = name
		attendance.distances = np.mean(dist)
		attendance.sites = site
		attendance.observation = observation
		session.add(attendance)
		if echo:
			print(attendance.__repr__())
	session.commit()

	# q

class Colour:
	# BGR 
	red = (0,0,255)
	green = (0,255,0)
	blue = (255,0,00)
	purple = (255,0,255)
	white = (255,255,255)
	black = (0,0,0)


def encode_image(img):	
	return face_recognition.face_encodings(img)[0]


def load_reference_images(path='reference_images'):
	print('loading reference images')
	names =[]
	encodings = []
	path = Path(path)
	lof = list(path.glob('*.png')) + list(path.glob('*.jpg'))
	for img_path in lof:
		name = img_path.stem.title()
		print(f'Encoding: {name}')
		img = face_recognition.load_image_file(str(img_path))
	## convert to RGB 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encodings.append(encode_image(img))
		names.append(name)
	if len(encodings)==0:
		raise ValueError('No reference images found')
	return (names, encodings)

def draw_box(frame, coords1, coords2, colour, thickness=2, name='Unknown'):
	
	x1,y1 = coords1
	x2,y2 = coords2
	cv2.rectangle(frame, coords1 ,coords2, colour, thickness)
	cv2.rectangle(frame, (x1, y2-35) , coords2 , colour, cv2.FILLED)
	cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, Colour.white ,2)


def main():

	record_invertal = 1
	site = 'NYC'

	CWD = Path.cwd() 
	reference_path = CWD / 'reference_images'
	db_dir = CWD / 'database'
	if not db_dir.exists():
		db_dir.mkdir(parent=True)
	names, ref_encodings =load_reference_images(path=reference_path)

	video_stream = cv2.VideoCapture(0)
	time.sleep(2.0)
	scale_frame = 0.25
	

	current_time  =  datetime.datetime.now()
	discard = current_time + datetime.timedelta(minutes=record_invertal)
	data = defaultdict(list) 

	echo = True

	while True:

		## grab frames 
		res, frame = video_stream.read()
		
		# exit()
		# if res:
			## resize the video image to 1/4 of size
		imgs = cv2.resize(frame, (0,0), None, scale_frame, scale_frame)
		## BGR ==> RGB
		imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

		faces_locations = face_recognition.face_locations(imgs)
		encode_face_frame = face_recognition.face_encodings(imgs, known_face_locations=faces_locations)
		## iter over all face and locations
		for encoded_face, face_loc in zip(encode_face_frame,faces_locations):
			
			## list of face matches 
			matches = face_recognition.compare_faces(ref_encodings, encoded_face)
			## get the eudi distances
			face_distance = face_recognition.face_distance(ref_encodings, encoded_face)
			# print(face_distance)
			if len(face_distance)==0:
				# continue
				break 
			## index of the min distance
			idx = np.argmin(face_distance)
			# print(face_loc)
			y1,x2, y2,x1 = [int(x/scale_frame) for x in face_loc]
			# print(y1,x2, y2,x1)
			name =names[idx]
			if matches[idx]:
				#name = names[idx]
				## scale down box size
				draw_box(frame, (x1,y1), (x2,y2), Colour.green, name=f'{name} dis:{face_distance[idx]:.2f}' )

			else:
			# y1,x2, y2,x1 = [x/scale_frame for x in face_loc]
				draw_box(frame, (x1,y1), (x2,y2), Colour.red, name=f'Unknown dis: {face_distance[idx]:.2f}')
			data[name].append(face_distance[idx])
			# time.sleep(.5)


		current_time  =  datetime.datetime.now()
		if current_time >= discard:
			print("*"*50,'\n', data)
			discard = current_time + datetime.timedelta(minutes=record_invertal) ## reset timerr
			add_attendee(data, site, echo) 
			
			data = defaultdict(list) ## reset dict 

		key = cv2.waitKey(1) & 0xFF
  #   # if the `q` key was pressed, break from the loop
		if key == ord("q"):
			add_attendee(data, site, echo)
			session.close()
			break
	## this the full frame NOT resized 
		cv2.imshow('WebCam preview', frame)


# add_attendee(data, site, echo)
# session.close()
cv2.destroyAllWindows()



if __name__ == "__main__":


	main()
