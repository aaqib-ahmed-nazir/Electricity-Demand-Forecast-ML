
import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { DayPicker, useNavigation } from "react-day-picker";

import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export type CalendarProps = React.ComponentProps<typeof DayPicker>;

// Custom caption component that includes year selection dropdown
function CustomCaption(props: { 
  displayMonth: Date; 
  onChange?: (date: Date) => void;
}) {
  const { displayMonth, onChange } = props;
  const { goToMonth, nextMonth, previousMonth } = useNavigation();
  
  // Get current month and year
  const currentMonth = displayMonth.getMonth();
  const currentYear = displayMonth.getFullYear();
  
  // Define available years (2018-2020)
  const years = [2018, 2019, 2020];
  
  // Handle month navigation
  const handlePreviousMonth = () => {
    if (previousMonth) goToMonth(previousMonth);
  };
  
  const handleNextMonth = () => {
    if (nextMonth) goToMonth(nextMonth);
  };
  
  // Handle year selection
  const handleYearChange = (yearStr: string) => {
    const year = parseInt(yearStr, 10);
    const newDate = new Date(displayMonth);
    newDate.setFullYear(year);
    goToMonth(newDate);
  };
  
  const monthName = displayMonth.toLocaleString('default', { month: 'long' });
  
  return (
    <div className="flex justify-center pt-1 relative items-center">
      <div className="flex items-center justify-between w-full">
        <button
          type="button"
          onClick={handlePreviousMonth}
          disabled={!previousMonth}
          className={cn(
            buttonVariants({ variant: "outline" }),
            "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
          )}
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        
        <div className="flex items-center gap-1">
          <span className="text-sm font-medium">
            {monthName}
          </span>
          
          <Select
            value={currentYear.toString()}
            onValueChange={handleYearChange}
          >
            <SelectTrigger className="h-7 w-[70px] text-xs">
              <SelectValue placeholder={currentYear.toString()}>
                {currentYear}
              </SelectValue>
            </SelectTrigger>
            <SelectContent className="max-h-[200px] pointer-events-auto">
              {years.map((year) => (
                <SelectItem key={year} value={year.toString()}>
                  {year}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <button
          type="button"
          onClick={handleNextMonth}
          disabled={!nextMonth}
          className={cn(
            buttonVariants({ variant: "outline" }),
            "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
          )}
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function Calendar({
  className,
  classNames,
  showOutsideDays = true,
  fromYear = 2018,
  toYear = 2020,
  ...props
}: CalendarProps & { fromYear?: number; toYear?: number }) {
  return (
    <DayPicker
      showOutsideDays={showOutsideDays}
      className={cn("p-3 pointer-events-auto", className)}
      classNames={{
        months: "flex flex-col sm:flex-row space-y-4 sm:space-x-4 sm:space-y-0",
        month: "space-y-4",
        caption: "flex justify-center pt-1 relative items-center",
        caption_label: "text-sm font-medium",
        nav: "space-x-1 flex items-center",
        nav_button: cn(
          buttonVariants({ variant: "outline" }),
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
        ),
        nav_button_previous: "absolute left-1",
        nav_button_next: "absolute right-1",
        table: "w-full border-collapse space-y-1",
        head_row: "flex",
        head_cell:
          "text-muted-foreground rounded-md w-9 font-normal text-[0.8rem]",
        row: "flex w-full mt-2",
        cell: "h-9 w-9 text-center text-sm p-0 relative [&:has([aria-selected].day-range-end)]:rounded-r-md [&:has([aria-selected].day-outside)]:bg-accent/50 [&:has([aria-selected])]:bg-accent first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md focus-within:relative focus-within:z-20",
        day: cn(
          buttonVariants({ variant: "ghost" }),
          "h-9 w-9 p-0 font-normal aria-selected:opacity-100"
        ),
        day_range_end: "day-range-end",
        day_selected:
          "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
        day_today: "bg-accent text-accent-foreground",
        day_outside:
          "day-outside text-muted-foreground opacity-50 aria-selected:bg-accent/50 aria-selected:text-muted-foreground aria-selected:opacity-30",
        day_disabled: "text-muted-foreground opacity-50",
        day_range_middle:
          "aria-selected:bg-accent aria-selected:text-accent-foreground",
        day_hidden: "invisible",
        ...classNames,
      }}
      components={{
        Caption: (captionProps) => <CustomCaption {...captionProps} />,
      }}
      fromYear={fromYear}
      toYear={toYear}
      captionLayout="buttons"
      {...props}
    />
  );
}
Calendar.displayName = "Calendar";

export { Calendar };
