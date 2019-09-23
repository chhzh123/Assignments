-- Q1
select name
from instructor
where dept_name = 'Biology';

-- Q2
select title
from course
where credits = 3;

-- Q3
select course_id, title
from takes natural join course
where ID = 12345;

-- Q4
select sum(credits)
from takes natural join course
where ID = 12345;

-- Q5
select ID, sum(credits)
from takes natural join course
group by ID;

-- Q6
select distinct S.name
from student as S, 
	(select * from takes natural join course) as C
where S.ID = C.ID and C.dept_name = 'Comp. Sci.';

-- Q7 (without except!)
select ID
from instructor
where ID not in
	(select ID
		from instructor natural join teaches);

-- Q8
select name, ID
from instructor
where ID not in
	(select ID
		from instructor natural join teaches);