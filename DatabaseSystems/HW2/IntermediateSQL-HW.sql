-- Q1
-- Spring semester sometimes have two sectors, thus needs sec_id
select max(enrollment), min(enrollment)
from (select sec_id, semester, year, count(distinct id) as enrollment
	from takes
	group by sec_id, semester, year) as T;

-- Q2
with T(sec_id, semester, year, enrollment) as
	(select sec_id, semester, year, count(distinct id)
		from takes
		group by sec_id, semester, year)
select T.sec_id, T.semester, T.year, T.enrollment
from T, (select max(enrollment) as maxE, min(enrollment)
	from T) as res
where T.enrollment = res.maxE;

-- Test data (need to preserve the constraints)
delete from course
	where course_id = 'CS-001';
delete from section
	where sec_id = '1' and semester = 'Fall' and year = '2010';
insert into course(course_id)
	values ('CS-001');
insert into section(course_id, sec_id, semester, year)
	values ('CS-001','1','Fall','2010');
-- Q3.1
select distinct sec_id, semester, year,
	(select count(distinct id)
		from takes
		where (takes.sec_id, takes.semester, takes.year) = (section.sec_id, section.semester, section.year)) as cnt
from section;
-- Q3.2
select distinct sec_id, semester, year, ifnull(cnt,0)
from section left outer join
	(select sec_id, semester, year, count(distinct id) as cnt
	from takes
	group by sec_id, semester, year) as T
using (sec_id, semester, year);

-- Q4
select course_id, title
from section natural join course
where course_id like 'CS-1%';

-- Q5.1
select distinct ID, name
from (select * from teaches natural join instructor) as T
where not exists (select cs_course.course_id
				from (select course_id
						from course
						where course_id like 'CS-1%') as cs_course
				where cs_course.course_id not in
					(select course_id
						from (select * from teaches natural join instructor) as U
						where U.name = T.name));

-- Q5.2
with U(course_id) as -- all course_id like 'CS-1%'
	(select distinct course_id
		from teaches natural join instructor
		where course_id like 'CS-1%')
select distinct ID, name
from (select * from teaches natural join instructor) as T
where ((select count(distinct course_id)
		from teaches natural join instructor
		where name = T.name and course_id like 'CS-1%')
	= (select count(course_id) from U));