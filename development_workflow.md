
## Development Workflow

### A. Choose an issue to work on 
1. Choose an issue from the "Ready" column (or, if needed, create an issue)
    - https://github.com/orgs/opendifferentialprivacy/projects/1
2. Add yourself as the Assignee
3. Move the Issue to the Ready Column
    - Note the issue number, e.g. Issue `#1`


### B. Create a new branch from the develop branch
1. Name the new branch starting with the issue number and description of your choice
    - e.g. "1_" + "impute" ->
      - `1_impute`

### C. Switch your development machine to the new branch

- e.g. `git checkout 1_impute`

### D. Upate your branch with any changes on develop

1. Check in your changes to the new  branch, `1_impute`
1. Checkout the develop branch and retrieve any changes
    ```
    git checkout develop
    git pull
    ```
1. Checkout the dev branch (`1_impute`)
    ```
    # git checkout [dev branch]
    git checkout 1_impute
    ```
1. Merge and fix any conflicts
    ```
    git merge origin develop
    ```

### E. Ready for Review 

1. If needed, merge with develop (repeat step D)
1. Check that you've written/updated any needed tests
1. Make a pull request
1. Have a team member review the pull request
1. Merge the pull request!
