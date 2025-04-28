/**
 *
 */
export default function ModeSelector() {
	return (
		<select
			defaultValue="What would you like to do?"
			className="select w-full min-w-0 rounded-2xl"
		>
			<option disabled={true}>What would you like to do?</option>
			<option>Search</option>
			<option>Q & A</option>
			<option>Diagnosis</option>
		</select>
	);
}
