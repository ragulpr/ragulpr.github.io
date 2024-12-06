IMAGE_NAME = jekyll-site

.PHONY: serve build clean

serve: ## Run the Jekyll site
	docker run --rm -p 4000:4000 -v $(PWD):/site $(IMAGE_NAME)

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

clean: ## Clean up
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	rm -rf _site .jekyll-cache .sass-cache