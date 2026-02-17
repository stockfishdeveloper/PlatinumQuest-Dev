//-----------------------------------------------------------------------------
// Compile all scripts in the project
//-----------------------------------------------------------------------------

function compileAllScripts() {
	echo("Starting compilation of all scripts...");

	// Compile all .cs files
	%pattern = "*.cs";
	for (%file = findFirstFile(%pattern); %file !$= ""; %file = findNextFile(%pattern)) {
		if (strstr(%file, ".dso") == -1) {
			echo("Compiling: " @ %file);
			compile(%file);
		}
	}

	// Compile all .gui files
	%pattern = "*.gui";
	for (%file = findFirstFile(%pattern); %file !$= ""; %file = findNextFile(%pattern)) {
		if (strstr(%file, ".dso") == -1) {
			echo("Compiling: " @ %file);
			compile(%file);
		}
	}

	// Recursively compile platinum folder
	compileFolder("platinum");

	echo("Compilation complete!");
}

function compileFolder(%folder) {
	echo("Compiling folder: " @ %folder);

	// Compile .cs files
	%pattern = %folder @ "/*.cs";
	for (%file = findFirstFile(%pattern); %file !$= ""; %file = findNextFile(%pattern)) {
		if (strstr(%file, ".dso") == -1 && strstr(%file, "/dev/") == -1) {
			echo("  Compiling: " @ %file);
			compile(%file);
		}
	}

	// Compile .gui files
	%pattern = %folder @ "/*.gui";
	for (%file = findFirstFile(%pattern); %file !$= ""; %file = findNextFile(%pattern)) {
		if (strstr(%file, ".dso") == -1) {
			echo("  Compiling: " @ %file);
			compile(%file);
		}
	}

	// Recursively process subdirectories
	%pattern = %folder @ "/*";
	for (%file = findFirstFile(%pattern); %file !$= ""; %file = findNextFile(%pattern)) {
		if (fileExt(%file) $= "" && fileName(%file) !$= "dev") {
			// This is a directory, recurse into it
			compileFolder(%file);
		}
	}
}

// Run the compilation
compileAllScripts();
