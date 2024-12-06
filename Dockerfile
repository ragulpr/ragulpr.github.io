FROM ruby:3.3.4

# Install build essentials
RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /site

# Copy just the Gemfile
COPY Gemfile .

# Install gems
RUN bundle install

EXPOSE 4000

CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]