{
  description = "A Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Use the nixpkgs for the specified system
        pkgs = nixpkgs.legacyPackages.${system};

        # Specify the Python version and packages
        python = pkgs.python313;
        python-packages = with python.pkgs; [
          # Add your Python packages here
          requests
          pandas
          numpy
          google-genai
          python-dotenv
          fastapi
          uvicorn
          python-multipart
          pip
          aiofiles
          pydantic
          pymupdf
          pillow
          scikit-learn
          langchain
          langchain-community
          langchain-google-genai
          tiktoken
          chromadb
          psycopg
        ];
      in
      {
        # The development shell
        devShells.default = pkgs.mkShell {
          name = "python-project-shell";

          # Packages available in the shell (including Python itself and other tools)
          packages = [
            python
          ] ++ python-packages;

          # Optional: Environment variables to set
          shellHook = ''
            echo "Entered Python dev shell! 🐍"
            export PYTHONIOENCODING=UTF-8
          '';
        };
      });
}