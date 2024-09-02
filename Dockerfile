# Base image
FROM ruby:3.3

# Update package list and install necessary packages
RUN apt-get update -qq && apt-get install -y build-essential wget unzip

# Install a specific version of RubyGems
RUN wget https://rubygems.org/rubygems/rubygems-3.5.17.zip && \
    unzip rubygems-3.5.17.zip && \
    cd rubygems-3.5.17 && \
    ruby setup.rb && \
    cd .. && \
    rm -rf rubygems-3.5.17.zip rubygems-3.5.17

# Install additional Ruby packages
RUN apt-get install -y ruby-full

# Set the working directory
WORKDIR /usr/src/app

# Copy Gemfile and Gemfile.lock to the container first
COPY Gemfile Gemfile.lock /usr/src/app/

# Install bundler and dependencies
RUN gem install bundler && bundle install

# Expose port 4000 to the host
EXPOSE 4000

# Clone the Jekyll blog repository
COPY . /usr/src/app

# Start the Jekyll server
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]
