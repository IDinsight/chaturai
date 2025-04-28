export default {
	extends: ["stylelint-config-standard", "stylelint-config-tailwindcss"],
	reportInvalidScopeDisables: true,
	reportNeedlessDisables: true,
	rules: {
		"block-no-empty": true,
		"color-no-invalid-hex": true,
		"font-weight-notation": "numeric",
		"property-no-unknown": true,
		"unit-allowed-list": ["em", "rem", "%", "s", "px", "fr"],
	},
};
