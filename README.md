# flask-container

A skeleton Python3 / Flask container template for projects.

## How To Use

- Your app logic should register back to `app.py` as Flask paths and methods.
- Add any Python dependencies to `requirements.txt`

## Build the Container

From within your cloned and extended version of this repository, first build the container:

```
docker build -t your-container-name .
```

Next you can run the container locally to test:

```
docker run -d -p 8080:80 your-container-name
```

Once that runs (check its status with `docker ps`) you can see the Flask site by visiting http://localhost:8080/ in your browser.

Finally, commit and push your code. You can also push your container image to Docker Hub or another registry:

```
docker push your-org/your-container-name
```

## Launching in DCOS @ UVA

1. Work with UVA Research Computing to set up your initial Flask service. This may include "production" and "dev" sites along with your `master` and `dev` branches in Git.
2. We recommend using a CI/CD tool such as Travis-CI to build and deploy your application container with each `git push`. See the example `.travis.yml` file for a basic build+push.
3. UVARC can integrate your build so that it is launched in DCOS as part of that build process.
