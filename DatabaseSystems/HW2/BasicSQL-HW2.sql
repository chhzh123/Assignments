drop table actor_role;
drop table actors;
drop table movies;

-- Q9
create table actors
	(AID    varchar(20),
	 name   varchar(50),
	 primary key (AID));

create table movies
	(MID    varchar(20),
	 title  varchar(50),
	 primary key (MID));

create table actor_role
	(MID    varchar(20),
	 AID    varchar(20),
	 rolename varchar(30),
	 primary key (MID,AID,rolename),
	 foreign key (MID) references movies(MID),
	 foreign key (AID) references actors(AID));

-- Q10
delete from actor_role;
delete from actors;
delete from movies;
insert into actors values ('01','Charlie Chaplin');
insert into movies values ('M1','Modern Times');
insert into actor_role values ('M1','01','Worker');
insert into movies values ('M2','The Great Dictator');
insert into actor_role values ('M2','01','Adenoid');
insert into actor_role values ('M2','01','Barber');
insert into movies values ('M3','City Lights');
insert into actor_role values ('M3','01','Tramp');
insert into actors values ('02','Leslie Cheung');
insert into movies values ('M4','Farewell My Concubine');
insert into actor_role values ('M4','02','Dieyi Cheng');
insert into actors values ('03','Tom Hanks');
insert into movies values ('M5','Forrest Gump');
insert into actor_role values ('M5','03','Gump');
insert into actors values ('04','No-movie Actor');

-- Q11
select name, title, count(rolename)
from actor_role natural join movies natural join actors
where name = 'Charlie Chaplin'
group by MID;

-- Q12
drop view no_movie_actor;
create view no_movie_actor as
select name
from actors
where name not in (select distinct name
					from actor_role natural join actors);
select * from no_movie_actor;

-- Q13
select *
from ((select name, title
		from actors, movies, actor_role
		where actors.AID = actor_role.AID and movies.MID = actor_role.MID)
	union
	(select name as name, null as title
		from no_movie_actor))
	as result;