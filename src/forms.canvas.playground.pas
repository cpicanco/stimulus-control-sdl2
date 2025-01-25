{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.canvas.playground;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtDlgs, Menus;

type

  { TFormCanvas }

  TFormCanvas = class(TForm)
    MenuItemDrawEquivalenceClasses: TMenuItem;
    PopupMenuSave: TPopupMenu;
    SavePictureDialog1: TSavePictureDialog;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure MenuItemDrawEquivalenceClassesClick(Sender: TObject);
  protected
    procedure Paint; override;
  private

  public

  end;

var
  FormCanvas: TFormCanvas;

implementation

{$R *.lfm}

uses picanco.experiments.output;

{ TFormCanvas }

procedure TFormCanvas.MenuItemDrawEquivalenceClassesClick(Sender: TObject);
begin
  if SavePictureDialog1.Execute then begin
    Bitmap.SaveToFile(SavePictureDialog1.FileName);
  end;
end;

procedure TFormCanvas.Paint;
begin
  inherited Paint;
  Canvas.Draw(0, 0, Bitmap);
end;

procedure TFormCanvas.FormCreate(Sender: TObject);
begin
  InitializeBitmap;
  VertScrollBar.Range := Bitmap.Height;
  HorzScrollBar.Range := Bitmap.Width;
end;

procedure TFormCanvas.FormDestroy(Sender: TObject);
begin
  FinalizeBitmap;
end;

end.

