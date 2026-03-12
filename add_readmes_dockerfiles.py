import os

def create_readme_if_missing(folder):
    readme_path = os.path.join(folder, "README.md")
    if not os.path.exists(readme_path):
        files = os.listdir(folder)
        content = f"# {os.path.basename(folder)}\n\nThis folder contains the following files and directories:\n"
        for f in files:
            content += f"- `{f}`\n"
        with open(readme_path, "w") as f:
            f.write(content)
        return True
    return False

def create_dockerfile_if_missing(folder, has_py, has_r):
    dockerfile_path = os.path.join(folder, "Dockerfile")
    if not os.path.exists(dockerfile_path):
        content = "FROM "
        if has_r and has_py:
            content += "rocker/r-base:latest\nRUN apt-get update && apt-get install -y python3 python3-pip\n"
        elif has_r:
            content += "rocker/r-base:latest\n"
        elif has_py:
            content += "python:3.10-slim\n"
        
        content += "WORKDIR /app\nCOPY . .\n"
        if has_py:
            content += "# RUN pip install -r requirements.txt # Un-comment if requirements.txt added\n"
            content += 'CMD ["bash"]\n'
        elif has_r:
            content += 'CMD ["bash"]\n'
            
        with open(dockerfile_path, "w") as f:
            f.write(content)
        
        # update readme to mention docker
        readme_path = os.path.join(folder, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "a") as f:
                f.write("\n## Docker Setup\n\nA Dockerfile is included in this directory to run the scripts in a containerized environment. Build the image with:\n`docker build -t image-name .`\nRun it with:\n`docker run -it image-name`\n")
        return True
    return False

def main():
    base_dir = "."
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(d) and not d.startswith(".")]
    
    updates = []
    
    for d in dirs:
        has_py = any(f.endswith(".py") for f in os.listdir(d))
        has_r = any(f.endswith(".R") for f in os.listdir(d))
        
        readme_created = create_readme_if_missing(d)
        docker_created = False
        if has_py or has_r:
            docker_created = create_dockerfile_if_missing(d, has_py, has_r)
            
        if readme_created or docker_created:
            updates.append(f"- **{d}**: " + ("] Added README. " if readme_created else "") + ("Added Dockerfile." if docker_created else ""))

    # Update main
    if updates:
        with open("README.md", "a") as f:
            f.write("\n\n## Recent Workspace Updates\n")
            for u in updates:
                f.write(f"{u}\n")
    print("Done")

if __name__ == "__main__":
    main()
