"use client";

import {useEffect} from "react";

import {themeChange} from "theme-change";

/**
 *
 */
export default function ThemeSelect() {
	useEffect(() => {
		themeChange(false); // false = don't watch for mutations
	}, []);

	return (
		<select
			className="select w-full min-w-16 truncate rounded-xl border-double transition-all duration-200"
			data-choose-theme
		>
			<option value="coffee">Coffee</option>
			<option value="dark">Dark</option>
			<option value="dim">Dim</option>
			<option value="forest">Forest</option>
			<option value="lemonade">Lemonade</option>
			<option value="light">Light</option>
			<option value="night">Night</option>
			<option value="retro">Retro</option>
			<option value="sunset">Sunset</option>
			<option value="winter">Winter</option>
		</select>
	);
}
