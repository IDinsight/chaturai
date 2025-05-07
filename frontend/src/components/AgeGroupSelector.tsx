/**
 *
 */
export default function AgeGroupSelector() {
	return (
		<select
			defaultValue="Select patient's age group"
			className="select w-full min-w-0 rounded-2xl"
		>
			<option disabled={true}>Select patient's age group</option>
			<option>Birth to 2 Months</option>
			<option>2 Months to 5 Years</option>
		</select>
	);
}
