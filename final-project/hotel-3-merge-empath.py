import pandas
import pickle

# true_df = pickle.load(open('./input/true_emp_vec.p','rb'))

emp_arr = []
for a in range(52):
	emp_arr.append(pickle.load(open('./input/unlabelled_empmp_vec'+str(a)+'.p','rb')))

empmp_df = emp_arr[0]
for a in range(1,len(emp_arr)):
	empmp_df = empmp_df.append(emp_arr[a], ignore_index=True)

# print(true_df.equals(empmp_df))
pickle.dump(empmp_df, open('./input/unlabelled_empmp_vec.p','wb'))