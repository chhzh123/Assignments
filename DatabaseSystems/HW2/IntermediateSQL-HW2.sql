-- Q6
insert into student
	select ID, name, dept_name, '0'
	from instructor;

-- Q7
delete from student
	where (ID, name, dept_name) in
		(select ID, name, dept_name
			from instructor);

-- Q8
update student as S
set tot_cred = (
	select sum(credits)
	from takes natural join course
	where S.ID = takes.ID and takes.grade is not null);

-- Q9
update instructor as I
set salary = 10000 * (
	select count(distinct sec_id, semester, year)
	from teaches as T
	where I.ID = T.ID);

-- Q10
-- Number of courses taught by instuctors of CS department registered by the students
select count(cs_course.course_id)
from ((select distinct course_id
		from instructor natural join teaches
		where dept_name = 'Comp. Sci.') as cs_course
	inner join
	(select distinct course_id from takes) as all_course
	on cs_course.course_id = all_course.course_id);