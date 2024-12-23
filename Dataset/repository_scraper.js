const { Octokit } = require("@octokit/rest");
const fs = require("fs/promises");

/**
 * Place your own github token here.
 */
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
});

/**
 * This function shows the remaining amount of API requests you can perform + reset time.
 * An authenticated user can do 30 requests, otherwise you can do only 10.
 * If remaining == 0 => limit!
 */
async function getRateLimit() {
  try {
    const response = await octokit.request("GET /rate_limit");
    const { search } = response.data.resources;
    return {
      remaining: search.remaining,
      reset: search.reset,
    };
  } catch (error) {
    console.log(error);
    return { remaining: 0, reset: 0 };
  }
}

/**
 * This function calculates how much time to wait for the limit to reset.
 */
async function waitForRateLimitReset() {
  const { remaining, reset } = await getRateLimit();
  const resetTime = new Date(reset * 1000);
  const now = new Date();

  if (resetTime > now) {
    const waitTime = resetTime - now + 7500;
    console.log(`Waiting for reset at: ${resetTime}`);
    return new Promise((resolve) => setTimeout(resolve, waitTime));
  }
}

/**
 * This function introduces a delay.
 * 'ms' = the duration in milliseconds (integer)
 */
async function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * This function generates a list with all months between 2001 and April 2024.
 * Each month has 2 entries (exept for first), one with the first day (01), another with the second day (02)
 * In order to avoid overlaps when we limit our search space
 */

function generateDates() {
  let dateArray = [];
  for (let i = 2000; i <= 2023; i++) {
    for (let j = 1; j <= 12; j++) {
      if (j < 10) {
        dateArray.push(`${i}-0${j}-01`);
        dateArray.push(`${i}-0${j}-02`);
      } else {
        dateArray.push(`${i}-${j}-01`);
        dateArray.push(`${i}-${j}-02`);
      }
    }
  }

  for (let j = 1; j <= 4; j++) {
    if (j < 10) {
      dateArray.push(`2024-0${j}-01`);
      dateArray.push(`2024-0${j}-02`);
    } else {
      dateArray.push(`2024-${j}-01`);
      dateArray.push(`2024-${j}-02`);
    }
  }

  dateArray.splice(1, 1);
  return dateArray;
}

/**
 * This function extracts GitHub repositories.
 * 'language' = main language of repos (string - start with capital letter)
 * 'totalCount' = amount of repositories to extract (integer)
 */
async function* searchReposByLicense(language, totalCount) {
  const reposPerPage = 100;
  const totalPages = Math.ceil(totalCount / reposPerPage);
  let allRepos = 0;
  let repoCounter = 0;

  try {
    let up = 900;
    let down = 850;
    while (true) {
      let intervalRepoC = 0;
      let totalRepos = new Set();
      for (let page = 1; page <= 10; page++) {
        const { remaining, reset } = await getRateLimit();

        console.log(`Remaining: ${remaining}`);
        console.log(`Reset at: ${reset}`);

        if (remaining <= 0) await waitForRateLimitReset();

        let response = [];
        if (up === 900) {
          response = await octokit.request("GET /search/repositories", {
            q: `license:gpl-3.0 license:gpl-2.0 license:agpl-3.0 language:${language} stars:>=900`,
            per_page: reposPerPage,
            page: page,
          });
        } else {
          response = await octokit.request("GET /search/repositories", {
            q: `license:gpl-3.0 license:gpl-2.0 license:agpl-3.0 language:${language} stars:${down}..${up}`,
            per_page: reposPerPage,
            page: page,
          });
        }

        const uniqueRepos = new Set(response.data.items);
        if (uniqueRepos.size > 0) {
          totalRepos = new Set([...totalRepos, ...uniqueRepos]);
          intervalRepoC += uniqueRepos.size;
        } else {
          break;
        }
        await wait(2500)
      }

      if (intervalRepoC > 0 && intervalRepoC < 1000) {
        repoCounter += totalRepos.size;
        if (up === 900) {
          console.log("Star interval: >=900");
        } else {
          console.log(`Star interval: ${down}..${up}`);
        }
        console.log(`Accumulated Repos: ${repoCounter}`);

        for (const repo of totalRepos) {
          if (allRepos < totalCount) {
            yield repo;
            allRepos += 1;
          } else {
            console.log(allRepos);
            return;
          }
        }
        await wait(3500);
      } else if (intervalRepoC >= 1000) {
        console.log(`Over 1000 repos for star interval: ${down}..${up}`);
        const dateArray = generateDates();
        for (let date = 0; date <= dateArray.length - 2; date += 2) {
          for (let page = 1; page <= 10; page++) {
            const { remaining, reset } = await getRateLimit();

            console.log(`Remaining: ${remaining}`);
            console.log(`Reset at: ${reset}`);

            if (remaining <= 0) await waitForRateLimitReset();

            let response = [];
            if (up === 900) {
              response = await octokit.request("GET /search/repositories", {
                q: `license:gpl-3.0 license:gpl-2.0 license:agpl-3.0 language:${language} stars:>=900 created:${
                  dateArray[date]
                }..${dateArray[date + 1]}`,
                per_page: reposPerPage,
                page: page,
              });
            } else {
              response = await octokit.request("GET /search/repositories", {
                q: `license:gpl-3.0 license:gpl-2.0 license:agpl-3.0 language:${language} stars:${down}..${up} created:${
                  dateArray[date]
                }..${dateArray[date + 1]}`,
                per_page: reposPerPage,
                page: page,
              });
            }

            const uniqueRepos = new Set(response.data.items);
            if (uniqueRepos.size > 0) {
              if (up === 900) {
                console.log(
                  `Star interval: >=900 between ${dateArray[date]} - ${
                    dateArray[date + 1]
                  }`
                );
              } else {
                console.log(
                  `Star interval: ${down}..${up} between ${dateArray[date]} - ${
                    dateArray[date + 1]
                  }`
                );
              }
              repoCounter += uniqueRepos.size;
              console.log(`Accumulated Repos: ${repoCounter}`);

              for (const repo of uniqueRepos) {
                if (allRepos < totalCount) {
                  yield repo;
                  allRepos += 1;
                } else {
                  console.log(allRepos);
                  return;
                }
              }
            } else {
              break;
            }
          await wait(2500);
          }
        await wait(4500);
        }
      }

      if (allRepos < totalCount && up === down && down === 0) {
        console.log(`There are only ${allRepos} repos!`);
        return;
      }

      if (up === 900) {
        up--;
      } else if (down > 100) {
        up -= 50;
        down -= 50;
      } else if (down === 100 && up === 149) {
        down -= 10;
        up -= 50;
      } else if (down > 10 && down < 100) {
        down -= 10;
        up -= 10;
      } else {
        down--;
        up = down;
      }
    }
  } catch (error) {
    console.log(error);
  }
}

/**
 * This function extracts specific fields for each repository and saves them in a json file.
 * 'languages' = languages to extract repos from (array of strings - start with capital letter)
 * 'repoC' = total amount of repos to extract (integer)
 */
async function saveRepos(languages, repoC) {
  const fields = [
    "id",
    "full_name",
    "html_url",
    "stargazers_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "language",
    "created_at",
    "pushed_at",
    "license",
  ];
  try {
    let repoAmount = repoC;
    let filePathGlb = "";
    let jsonRepos = "";
    for (lang of languages) {
      let counter = 0;
      filePathGlb = `./your_path/${lang}StrongCopyLeft10500.json`;
      await fs.writeFile(filePathGlb, "[", "utf8", (err) => {
        if (err) {
          consolge.error("Error opening JSON array:", err);
        }
      });
      const repoGen = searchReposByLicense(lang, repoAmount);
      for await (const repo of repoGen) {
        const mappedRepo = {};
        fields.forEach((field) => {
          if (repo.hasOwnProperty(field)) {
            mappedRepo[field] = repo[field];
          }
        });
        if (Object.keys(mappedRepo).length !== 0) {
          const date = new Date().toLocaleString("en-US", {
            timeZone: "Europe/Amsterdam",
          });
          const dateTimeWithTimeZone = `${date} (Europe/Amsterdam)`;
          mappedRepo.retrieval_date = dateTimeWithTimeZone;
          jsonRepos = JSON.stringify(mappedRepo);
          if (counter < repoAmount - 1) {
            await fs.appendFile(filePathGlb, jsonRepos + ",", "utf8");
          }
          counter++;
        }
      }
      console.log(counter);
      await fs.appendFile(filePathGlb, jsonRepos + "]", "utf8");
      await wait(180000);
    }
  } catch (error) {
    console.error("Error parsing JSON:", error);
  }
}

function main() {
  const langs = process.argv.slice(2);
  const repos = langs.pop();
  const repoC = parseInt(repos);
  saveRepos(langs, repoC);
}

main();
